//! Native macOS ARM64 User-Space GA102 (Ampere sm_86) Hardware Driver & Multi-Kernel `.cubin` Loader.
//! Speaks directly to `TinyGpuClient` (`tinygpu.sock` + `MAP_SYSMEM_FD`), boots GSP-RM from
//! pre-cached binary firmware, manages 5-level MMU v2 page tables (with 2 MB huge pages and
//! batched BAR1 shadow-table flushes), and drives Ampere Compute (`0xc7c0`) & Copy (`0xc7b5`) GPFIFOs.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{fence, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::nv_defs::*;
use crate::tinygpu_client::{SysmemMapping, TinyGpuClient};

#[inline]
pub const fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

#[inline]
pub const fn align_down(val: u64, align: u64) -> u64 {
    val & !(align - 1)
}

// -----------------------------------------------------------------------------
// GPU Buffer Types
// -----------------------------------------------------------------------------

pub struct VramBuffer {
    pub va_addr: u64,
    pub size: u64,
    pub paddrs: Vec<(u64, u64)>,
}

pub struct HostMappedBuffer {
    pub va_addr: u64,
    pub size: usize,
    pub mapping: SysmemMapping,
}

impl HostMappedBuffer {
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.mapping.ptr
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.mapping.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.mapping.as_mut_slice()
    }
}

// -----------------------------------------------------------------------------
// 5-Level Turing/Ampere MMU v2 Page Table Manager (`va_bits=48`, `[12,21,29,38,47]`)
// -----------------------------------------------------------------------------

const MMU_COVERS: [u64; 5] = [1 << 47, 1 << 38, 1 << 29, 1 << 21, 1 << 12];
const MMU_CNT: [usize; 5]  = [4, 512, 512, 256, 512]; // lv=3 is Dual-PDE (256 * 16B = 4096B)

struct PtPage {
    words: [u64; 512],
    dirty: bool,
}

pub struct VramMemoryManager {
    pub vram_size: u64,
    pub usable_vram: u64,
    pub root_pt_paddr: u64,
    boot_next: u64,
    boot_end: u64,
    pt_next: u64,
    pt_end: u64,
    bar1_pa_next: u64,
    bar1_pa_end: u64,
    high_pa_next: u64,
    high_pa_end: u64,
    va_next: u64,
    pts: HashMap<u64, PtPage>,
}

impl VramMemoryManager {
    pub fn new(vram_size: u64, bar1_size: u64) -> Self {
        // Reserve top 224 MiB strictly below GSP-RM's hardware-locked WPR2 region (~188 MiB:
        // 129 MiB gsp_heap + ~55 MiB gsp_image + FRTS + VGA + non_wpr) so user allocations never touch WPR2!
        let usable_vram = vram_size.saturating_sub(224 << 20);
        let boot_end = 2 << 20;
        let pt_size = align_up(usable_vram / 512, 1 << 20);
        let pt_end = boot_end + pt_size;
        let bar1_cap = bar1_size.min(usable_vram);

        let mut mm = Self {
            vram_size,
            usable_vram,
            root_pt_paddr: 0,
            boot_next: 0,
            boot_end,
            pt_next: boot_end,
            pt_end,
            // Keep BAR1-accessible allocations (`cpu_access` / `zero=true`) inside `[pt_end, 256MB)`
            // and large DMA-populated model/KV buffers in `[256MB, usable_vram)`!
            bar1_pa_next: pt_end,
            bar1_pa_end: bar1_cap,
            high_pa_next: bar1_cap,
            high_pa_end: usable_vram,
            va_next: 0x10_0000_0000,
            pts: HashMap::new(),
        };

        let root = mm.alloc_pt_page(true);
        mm.root_pt_paddr = root;
        mm
    }

    fn alloc_pt_page(&mut self, boot: bool) -> u64 {
        let paddr = if boot {
            let p = self.boot_next;
            assert!(p + 0x1000 <= self.boot_end, "Boot VRAM allocator exhausted");
            self.boot_next += 0x1000;
            p
        } else {
            let p = self.pt_next;
            assert!(p + 0x1000 <= self.pt_end, "Page-table BAR1 VRAM allocator exhausted");
            self.pt_next += 0x1000;
            p
        };
        self.pts.insert(
            paddr,
            PtPage {
                words: [0u64; 512],
                dirty: true,
            },
        );
        paddr
    }

    pub fn alloc_vaddr(&mut self, size: u64, align: u64) -> u64 {
        let pwr2 = if size == 0 {
            0x1000
        } else {
            1u64 << (63 - size.leading_zeros())
        };
        let eff_align = pwr2.max(align).max(0x1000);
        let va = align_up(self.va_next, eff_align);
        self.va_next = va + align_up(size, 0x1000);
        va
    }

    pub fn palloc_bar1(&mut self, pci: &mut TinyGpuClient, size: u64, align: u64, zero: bool) -> Result<u64, String> {
        let sz = align_up(size, 0x1000);
        let p = align_up(self.bar1_pa_next, align.max(0x1000));
        if p + sz > self.bar1_pa_end {
            return Err(format!(
                "BAR1 mappable VRAM exhausted requesting {:#x} bytes (used {:#x}/{:#x})",
                sz, p, self.bar1_pa_end
            ));
        }
        self.bar1_pa_next = p + sz;
        if zero {
            let zchunk = vec![0u8; (sz as usize).min(256 * 1024)];
            let mut off = 0u64;
            while off < sz {
                let n = ((sz - off) as usize).min(zchunk.len());
                pci.bulk_write(1, p + off, &zchunk[..n])?;
                off += n as u64;
            }
        }
        Ok(p)
    }

    pub fn palloc_vram(&mut self, size: u64, align: u64) -> Result<u64, String> {
        let sz = align_up(size, 0x1000);
        let p = align_up(self.high_pa_next, align.max(0x1000));
        if p + sz <= self.high_pa_end {
            self.high_pa_next = p + sz;
            return Ok(p);
        }
        // Fallback to remaining BAR1 space if high VRAM is nearly full
        let p_low = align_up(self.bar1_pa_next, align.max(0x1000));
        if p_low + sz <= self.bar1_pa_end {
            self.bar1_pa_next = p_low + sz;
            return Ok(p_low);
        }
        Err(format!(
            "GPU VRAM OOM requesting {:.2} MB (high used {:.2}/{:.2} GB)",
            sz as f64 / 1e6,
            (self.high_pa_next - self.bar1_pa_end) as f64 / 1e9,
            (self.high_pa_end - self.bar1_pa_end) as f64 / 1e9
        ))
    }

    #[inline]
    fn pt_idx(lv: usize, va: u64) -> usize {
        ((va / MMU_COVERS[lv]) as usize) % MMU_CNT[lv]
    }

    fn ensure_child_pt(&mut self, pt_paddr: u64, lv: usize, idx: usize) -> u64 {
        let is_dual = lv == 3;
        let (lo, hi) = {
            let page = self.pts.get(&pt_paddr).unwrap();
            if is_dual {
                (page.words[2 * idx], page.words[2 * idx + 1])
            } else {
                (page.words[idx], 0)
            }
        };

        // Check if valid PDE already exists
        let valid_pde = if is_dual {
            (lo & 1) == 0 && ((hi >> 1) & 0x3) != 0
        } else {
            (lo & 1) == 0 && ((lo >> 1) & 0x3) != 0
        };
        if valid_pde {
            return if is_dual {
                let ap = (hi >> 1) & 0x3;
                let mask = if ap == 1 { 0x01ff_ffff } else { (1u64 << 46) - 1 };
                ((hi >> 8) & mask) << 12
            } else {
                let ap = (lo >> 1) & 0x3;
                let mask = if ap == 1 { 0x01ff_ffff } else { (1u64 << 46) - 1 };
                ((lo >> 8) & mask) << 12
            };
        }

        assert!(
            (lo & 1) == 0,
            "MMU PDE collision with existing leaf PTE at lv={} idx={}",
            lv,
            idx
        );
        let child_paddr = self.alloc_pt_page(false);
        let page = self.pts.get_mut(&pt_paddr).unwrap();
        if is_dual {
            let (d_lo, d_hi) = encode_mmu_v2_dual_pde_small(child_paddr, true);
            page.words[2 * idx] = d_lo;
            page.words[2 * idx + 1] = d_hi;
        } else {
            page.words[idx] = encode_mmu_v2_pde(child_paddr, true);
        }
        page.dirty = true;
        child_paddr
    }

    fn write_leaf_pte(&mut self, pt_paddr: u64, lv: usize, idx: usize, paddr: u64, is_sys: bool, uncached: bool) {
        let pte = encode_mmu_v2_pte(paddr, is_sys, uncached, true);
        let page = self.pts.get_mut(&pt_paddr).unwrap();
        if lv == 3 {
            // Dual-PDE level (lv=3, 2MB huge page): 16B stride per entry!
            page.words[2 * idx] = pte;
            page.words[2 * idx + 1] = 0;
        } else {
            page.words[idx] = pte;
        }
        page.dirty = true;
    }

    /// Create page tables down to leaf for `va` of `size` and return `(lv, pt_paddr)` stack
    /// matching `MemoryManager.page_tables(res_va, res_sz)` for `COPY_SERVER_RESERVED_PDES`.
    pub fn reserve_pde_levels(&mut self, va: u64, size: u64) -> Vec<(usize, u64)> {
        let mut stack = Vec::new();
        let mut cur_pt = self.root_pt_paddr;
        for lv in 0..5 {
            stack.push((lv, cur_pt));
            let covers = MMU_COVERS[lv];
            let can_huge = lv >= 2 && (va & (covers - 1)) == 0 && covers <= size;
            if lv == 4 || can_huge {
                break;
            }
            let idx = Self::pt_idx(lv, va);
            cur_pt = self.ensure_child_pt(cur_pt, lv, idx);
        }
        stack
    }

    pub fn map_range(
        &mut self,
        pci: &mut TinyGpuClient,
        va_start: u64,
        paddrs: &[(u64, u64)],
        is_sys: bool,
        uncached: bool,
    ) -> Result<(), String> {
        let mut cur_va = va_start;
        for &(seg_pa, seg_sz) in paddrs {
            let mut off = 0u64;
            while off < seg_sz {
                let rem = seg_sz - off;
                let pa = seg_pa + off;
                // Use 512MB (lv=2) or 2MB (lv=3) huge PTE when VA, PA, and remaining size allow!
                let target_lv = if !is_sys
                    && rem >= MMU_COVERS[2]
                    && (cur_va & (MMU_COVERS[2] - 1)) == 0
                    && (pa & (MMU_COVERS[2] - 1)) == 0
                {
                    2
                } else if !is_sys
                    && rem >= MMU_COVERS[3]
                    && (cur_va & (MMU_COVERS[3] - 1)) == 0
                    && (pa & (MMU_COVERS[3] - 1)) == 0
                {
                    3
                } else {
                    4
                };

                let mut pt = self.root_pt_paddr;
                for lv in 0..target_lv {
                    let idx = Self::pt_idx(lv, cur_va);
                    pt = self.ensure_child_pt(pt, lv, idx);
                }
                let step = MMU_COVERS[target_lv];
                let idx = Self::pt_idx(target_lv, cur_va);
                let max_batch = ((rem / step) as usize).min(MMU_CNT[target_lv] - idx);
                for i in 0..max_batch {
                    self.write_leaf_pte(pt, target_lv, idx + i, pa + (i as u64) * step, is_sys, uncached);
                }
                let mapped_bytes = (max_batch as u64) * step;
                cur_va += mapped_bytes;
                off += mapped_bytes;
            }
        }
        self.flush_mmu(pci)
    }

    pub fn sync_pt_from_vram(&mut self, pci: &mut TinyGpuClient, pt_paddr: u64) -> Result<(), String> {
        let mut raw_page = [0u8; 4096];
        pci.bulk_read(1, pt_paddr, &mut raw_page)?;
        if let Some(page) = self.pts.get_mut(&pt_paddr) {
            for i in 0..512 {
                page.words[i] = get_u64(&raw_page, i * 8);
            }
            page.dirty = false;
        }
        Ok(())
    }

    pub fn flush_mmu(&mut self, pci: &mut TinyGpuClient) -> Result<(), String> {
        let mut flushed_any = false;
        let mut pt_addrs: Vec<u64> = self
            .pts
            .iter()
            .filter_map(|(&pa, p)| if p.dirty { Some(pa) } else { None })
            .collect();
        pt_addrs.sort_unstable();

        let mut raw_page = [0u8; 4096];
        for pa in pt_addrs {
            let page = self.pts.get_mut(&pa).unwrap();
            for (i, &w) in page.words.iter().enumerate() {
                raw_page[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
            }
            pci.bulk_write(1, pa, &raw_page)?;
            page.dirty = false;
            flushed_any = true;
        }
        if flushed_any {
            // Flush PDB + HUB TLB + GPC TLB (`(1<<0) | (1<<1) | (1<<6) | (1<<31)`)
            pci.wreg(NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE, 0x8000_0043)?;
        }
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// GSP Message Queue Ring & Firmware Boot State
// -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct FwsecDesc {
    stored_size: u32,
    pkc_data_off: u32,
    interface_off: u32,
    imem_pa: u32,
    imem_sz: u32,
    imem_va: u32,
    dmem_pa: u32,
    dmem_sz: u32,
    eng_id_mask: u16,
    ucode_id: u8,
}

struct GspState {
    queues_mem: SysmemMapping,
    cmd_q_off: usize,
    stat_q_off: usize,
    cmd_seq: u32,
    msg_size: usize,
    msg_count: u32,
    rm_args_mem: SysmemMapping,
    logbuf_mem: SysmemMapping,
    libos_args_mem: SysmemMapping,
    gsp_radix3_chunks: Vec<SysmemMapping>,
    gsp_sig_mem: SysmemMapping,
    rm_booter_mem: SysmemMapping,
    wpr_meta_mem: SysmemMapping,
    frts_img_mem: SysmemMapping,
    sec2_booter_mem: SysmemMapping,
    fwsec_desc: FwsecDesc,
    sec2_code_off: u32,
    sec2_code_sz: u32,
    sec2_data_off: u32,
    sec2_data_sz: u32,
    next_handle: u32,
    priv_root: u32,
    user_root: u32,
    user_device: u32,
    user_subdevice: u32,
    user_virtmem: u32,
    user_ctxshare: u32,
    method_sysmems: Vec<SysmemMapping>,
    grctx_sizes: Vec<(u16, u64, bool, bool, bool)>, // (buf_id, size, phys, virt, local)
    grctx_kept_bufs: Vec<VramBuffer>,
}

// -----------------------------------------------------------------------------
// GPFIFO Ring & Multi-Kernel `.cubin` Program Structures
// -----------------------------------------------------------------------------

pub struct GpFifoRing {
    pub bar1_ring_off: u64,
    pub bar1_gpput_off: u64,
    pub entries_count: u32,
    pub put_value: u32,
    pub token: u32,
}

pub struct GpuKernel {
    pub name: String,
    pub prog_addr: u64,
    pub prog_sz: usize,
    pub regs_usage: u32,
    pub shmem_usage: u32,
    pub lcmem_usage: u32,
    pub cbuf0_size: u32,
    pub cbuf0_template: Vec<u8>,
    pub extra_cbufs: Vec<(usize, u64, u32)>,
    pub qmd_template: QmdV3,
}

pub struct CubinModule {
    pub code_buf: VramBuffer,
    pub kernels: HashMap<String, GpuKernel>,
}

// -----------------------------------------------------------------------------
// Top-Level GA102 Hardware Context (`NvGpu`)
// -----------------------------------------------------------------------------

pub struct NvGpu {
    pub pci: TinyGpuClient,
    pub mm: VramMemoryManager,
    gsp: GspState,
    pub chip_id: u32,
    pub chip_name: String,
    pub sm_version: u32,
    pub sass_version: u32,
    pub active_sms: u32,
    pub num_gpcs: u32,
    pub num_tpc_per_gpc: u32,
    pub num_sm_per_tpc: u32,
    pub max_warps_per_sm: u32,
    pub shared_mem_window: u64,
    pub local_mem_window: u64,
    pub slm_per_thread: u32,
    shader_local_mem: Option<VramBuffer>,

    gpfifo_area: VramBuffer,
    pub compute_gpfifo: GpFifoRing,
    pub dma_gpfifo: GpFifoRing,

    cmdq_buf: HostMappedBuffer,
    cmdq_wptr_bytes: usize,
    kernargs_buf: HostMappedBuffer,
    pub kernargs_wptr: usize,

    pub timeline_buf: HostMappedBuffer,
    pub timeline_value: u64,
    dma_staging_buf: HostMappedBuffer,
    unloaded: bool,
}

impl NvGpu {
    pub fn init() -> Result<Self, String> {
        let mut last_err = String::new();
        for attempt in 0..5 {
            match Self::init_once() {
                Ok(gpu) => return Ok(gpu),
                Err(e) => {
                    last_err = e;
                    if attempt + 1 < 5 {
                        eprintln!(
                            "[qwen-nv] Retrying GPU init ({}/4) after transient PCIe ASPM wakeup ({})...",
                            attempt + 1,
                            last_err
                        );
                        thread::sleep(Duration::from_millis(500 * (attempt as u64 + 1)));
                    }
                }
            }
        }
        Err(last_err)
    }

    fn init_once() -> Result<Self, String> {
        let t0 = Instant::now();
        let mut pci = TinyGpuClient::connect()?;

        // Enable PCI Memory Space (0x2) first (do NOT enable Bus Master until WPR2 check passes,
        // per PCIe Spec §6.6.2: FLR with active Bus Master / Falcon DMA can Surprise-Down Thunderbolt!)
        let pci_cmd = pci.read_config(PCI_COMMAND, 2)?;
        pci.write_config(PCI_COMMAND, (pci_cmd | 0x2) & !PCI_COMMAND_MASTER, 2)?;

        // If WPR2 is still locked from an unclean prior process, quiesce Falcons & DMA then issue PCI reset!
        let wpr2_hi = pci.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI)?;
        if wpr2_hi != 0 {
            eprintln!("[qwen-nv] WPR2 active ({:#x}); quiescing Falcons & issuing clean PCI reset...", wpr2_hi);
            Self::emergency_halt_falcons(&mut pci);
            let cmd_pre = pci.read_config(PCI_COMMAND, 2).unwrap_or(0);
            let _ = pci.write_config(PCI_COMMAND, cmd_pre & !PCI_COMMAND_MASTER, 2);
            thread::sleep(Duration::from_millis(25));
            pci.reset_pci()?;
            thread::sleep(Duration::from_millis(650));
            pci.remap_bars()?;
            let pci_cmd2 = pci.read_config(PCI_COMMAND, 2)?;
            pci.write_config(PCI_COMMAND, (pci_cmd2 | 0x2) & !PCI_COMMAND_MASTER, 2)?;
            Self::wait_for_gfw_boot(&mut pci)?;
            pci.remap_bars()?;
        }

        // Now enable PCI Memory Space (0x2) + Bus Master (0x4) cleanly
        let pci_cmd_final = pci.read_config(PCI_COMMAND, 2)?;
        pci.write_config(PCI_COMMAND, pci_cmd_final | PCI_COMMAND_MASTER | 0x2, 2)?;

        let chip_id = pci.rreg(NV_PMC_BOOT_0)?;
        let boot42 = pci.rreg(NV_PMC_BOOT_42)?;
        let arch_id = (boot42 >> 24) & 0x3f;
        let impl_id = (boot42 >> 20) & 0x0f;
        let arch_prefix = match arch_id {
            0x17 => "GA1",
            0x19 => "AD1",
            0x1b => "GB2",
            _ => return Err(format!("Unsupported NVIDIA architecture code {:#x} in BOOT_42={:#x}", arch_id, boot42)),
        };
        let chip_name = format!("{}{:02}", arch_prefix, impl_id);

        let vram_mb = pci.rreg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_42)? as u64;
        let vram_size = vram_mb << 20;
        let (_bar0_phys, _) = pci.bar_info(0)?;
        let (_bar1_phys, bar1_size) = pci.bar_info(1)?;

        let mut mm = VramMemoryManager::new(vram_size, bar1_size);

        // Prepare VBIOS FWSEC FRTS + Cached GSP Firmware in DART-mapped host memory
        let mut gsp = Self::prepare_gsp_sw(&mut pci, vram_size)?;

        // Boot GSP-RM via Falcon (FRTS) + SEC2 (Booter)
        if let Err(e) = Self::boot_falcons_and_gsp(&mut pci, &mut gsp) {
            Self::emergency_halt_falcons(&mut pci);
            return Err(e);
        }

        // From this point on, GSP-RM holds DART IOVAs: if ANY setup step fails, we MUST unload GSP
        // and halt Falcon/RISC-V DMA BEFORE `pci` drops and `TinyGPU` calls `IODMACommand::CompleteDMA`!
        let init_after_gsp = (|| -> Result<_, String> {
            // Clear BAR1 block FIRST so BAR1 accesses physical VRAM directly before flushing root PDE!
            pci.wreg(NV_PBUS_BAR1_BLOCK, 0)?;
            mm.flush_mmu(&mut pci)?;

            // Initialize GSP Golden Context Image
            Self::init_golden_image(&mut pci, &mut mm, &mut gsp)?;

            // Allocate User Root Client (`0xc1000000`) & Device / Subdevice / VASpace / GPFIFOs
            let user_root = gsp.user_root;
            let root_params = [0u8; 120];
            Self::rpc_rm_alloc(&mut pci, &mut mm, &mut gsp, 0, NV01_ROOT, &root_params, user_root)?;

            let mut dev_params = [0u8; 56];
            put_u32(&mut dev_params, 0, 0); // deviceId = 0
            put_u32(&mut dev_params, 4, user_root); // hClientShare
            put_u32(&mut dev_params, 48, 0); // NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES = 0
            let user_device = Self::rpc_rm_alloc(&mut pci, &mut mm, &mut gsp, user_root, NV01_DEVICE_0, &dev_params, user_root)?;
            gsp.user_device = user_device;

            let subdev_params = [0u8; 4];
            let user_subdevice = Self::rpc_rm_alloc(&mut pci, &mut mm, &mut gsp, user_device, NV20_SUBDEVICE_0, &subdev_params, user_root)?;
            gsp.user_subdevice = user_subdevice;

            let mut virt_params = [0u8; 24];
            put_u64(&mut virt_params, 8, 0x0001_ffff_ffff_ffff);
            let virtmem = Self::rpc_rm_alloc(&mut pci, &mut mm, &mut gsp, user_device, NV01_MEMORY_VIRTUAL, &virt_params, user_root)?;
            gsp.user_virtmem = virtmem;

            if std::env::var("QWEN_NV_MAX_BOOST").ok().as_deref() == Some("1") {
                let mut boost_params = [0u8; 8];
                put_u32(&mut boost_params, 0, NV2080_PERF_BOOST_FLAGS_MAX);
                put_u32(&mut boost_params, 4, 0xffff_ffff);
                let _ = Self::rpc_rm_control(&mut pci, &mut gsp, user_subdevice, NV2080_CTRL_CMD_PERF_BOOST, &boost_params, user_root)?;
            }

            // Allocate User VASpace (`FERMI_VASPACE_A`) & bind `mm.root_pt_paddr`
            let mut va_params = [0u8; 48];
            put_u32(&mut va_params, 4, NV_VASPACE_FLAGS_USER);
            put_u64(&mut va_params, 8, 0x0001_ffff_fb00_0000);
            put_u64(&mut va_params, 40, 0x1000);
            let user_vaspace = Self::rpc_rm_alloc(&mut pci, &mut mm, &mut gsp, user_device, FERMI_VASPACE_A, &va_params, user_root)?;
            Self::rpc_set_page_directory(&mut pci, &mut gsp, user_device, user_vaspace, mm.root_pt_paddr, user_root)?;

            // Channel Group (`KEPLER_CHANNEL_GROUP_A`)
            let mut chgrp_params = [0u8; 20];
            put_u32(&mut chgrp_params, 12, NV2080_ENGINE_TYPE_GRAPHICS);
            let channel_group = Self::rpc_rm_alloc(&mut pci, &mut mm, &mut gsp, user_device, KEPLER_CHANNEL_GROUP_A, &chgrp_params, user_root)?;

            // Allocate `gpfifo_area` (`3 MB` in BAR1-accessible VRAM)
            let gpfifo_pa = mm.palloc_bar1(&mut pci, 0x30_0000, 0x10000, true)?;
            let gpfifo_va = mm.alloc_vaddr(0x30_0000, 0x10000);
            mm.map_range(&mut pci, gpfifo_va, &[(gpfifo_pa, 0x30_0000)], false, false)?;
            let gpfifo_area = VramBuffer {
                va_addr: gpfifo_va,
                size: 0x30_0000,
                paddrs: vec![(gpfifo_pa, 0x30_0000)],
            };

            // Context Share (`FERMI_CONTEXT_SHARE_A`)
            let mut ctxshare_params = [0u8; 12];
            put_u32(&mut ctxshare_params, 0, user_vaspace);
            put_u32(&mut ctxshare_params, 4, NV_CTXSHARE_FLAGS_ASYNC);
            let ctxshare = Self::rpc_rm_alloc(&mut pci, &mut mm, &mut gsp, channel_group, FERMI_CONTEXT_SHARE_A, &ctxshare_params, user_root)?;
            gsp.user_ctxshare = ctxshare;

            // Create Compute GPFIFO (offset 0) & Copy/DMA GPFIFO (offset 0x100000)
            let compute_gpfifo = Self::create_gpfifo(
                &mut pci, &mut mm, &mut gsp, &gpfifo_area, ctxshare, channel_group, 0, 0x10000, true,
            )?;
            let dma_gpfifo = Self::create_gpfifo(
                &mut pci, &mut mm, &mut gsp, &gpfifo_area, ctxshare, channel_group, 0x10_0000, 0x10000, false,
            )?;

            // Schedule Channel Group (`NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS` is 2 bytes!)
            let mut sched_params = [0u8; 2];
            sched_params[0] = 1;
            let _ = Self::rpc_rm_control(&mut pci, &mut gsp, channel_group, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, &sched_params, user_root)?;

            // Query GPU SM / GPC / TPC topology dynamically from GSP-RM
            let gr_info_req = vec![0u8; 3712];
            let gr_info_resp = Self::rpc_rm_control(&mut pci, &mut gsp, user_subdevice, NV2080_CTRL_CMD_KGR_GET_INFO, &gr_info_req, user_root)?;
            let read_gr_info = |idx: usize| -> u32 { get_u32(&gr_info_resp, idx * 8 + 4) };
            let sm_version = read_gr_info(GR_INFO_SM_VERSION);
            let max_warps_per_sm = read_gr_info(GR_INFO_MAX_WARPS_PER_SM);
            let num_gpcs = read_gr_info(GR_INFO_NUM_GPCS);
            let num_tpc_per_gpc = read_gr_info(GR_INFO_NUM_TPC_PER_GPC);
            let num_sm_per_tpc = read_gr_info(GR_INFO_NUM_SM_PER_TPC).max(2);
            let sass_version = ((sm_version & 0xf00) >> 4) | (sm_version & 0x0f);

            let mut active_tpcs = 0u32;
            for gpc_id in 0..num_gpcs {
                let mut tpc_req = [0u8; 24];
                put_u32(&mut tpc_req, 16, gpc_id);
                if let Ok(tpc_resp) = Self::rpc_rm_control(&mut pci, &mut gsp, user_subdevice, NV2080_CTRL_CMD_GR_GET_TPC_MASK, &tpc_req, user_root) {
                    let mask = get_u32(&tpc_resp, 20);
                    active_tpcs += mask.count_ones();
                }
            }
            let active_sms = if active_tpcs > 0 { active_tpcs * num_sm_per_tpc } else { 68 };

            let cmdq_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 2 << 20)?;
            let kernargs_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 4 << 20)?;
            let timeline_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 0x1000)?;
            let dma_staging_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 4 << 20)?;

            Ok((
                sm_version,
                sass_version,
                active_sms,
                num_gpcs,
                num_tpc_per_gpc,
                num_sm_per_tpc,
                max_warps_per_sm,
                gpfifo_area,
                compute_gpfifo,
                dma_gpfifo,
                cmdq_buf,
                kernargs_buf,
                timeline_buf,
                dma_staging_buf,
            ))
        })();

        let (
            sm_version,
            sass_version,
            active_sms,
            num_gpcs,
            num_tpc_per_gpc,
            num_sm_per_tpc,
            max_warps_per_sm,
            gpfifo_area,
            compute_gpfifo,
            dma_gpfifo,
            cmdq_buf,
            kernargs_buf,
            timeline_buf,
            dma_staging_buf,
        ) = match init_after_gsp {
            Ok(tup) => tup,
            Err(e) => {
                // Unload GSP-RM / halt Falcon DMA before TinyGpuClient drops!
                let mut unload_req = [0u8; 8];
                put_u32(&mut unload_req, 4, 1 << 6);
                let unloaded_ok = Self::gsp_send_rpc(&mut pci, &mut gsp, NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, &unload_req)
                    .and_then(|_| Self::gsp_wait_resp(&mut pci, &mut gsp, NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, Duration::from_secs(2)))
                    .is_ok();
                if !unloaded_ok {
                    Self::emergency_halt_falcons(&mut pci);
                }
                thread::sleep(Duration::from_millis(20));
                return Err(e);
            }
        };

        let mut gpu = Self {
            pci,
            mm,
            gsp,
            chip_id,
            chip_name,
            sm_version,
            sass_version,
            active_sms,
            num_gpcs,
            num_tpc_per_gpc,
            num_sm_per_tpc,
            max_warps_per_sm,
            shared_mem_window: 0x7294_0000_0000,
            local_mem_window: 0x7293_0000_0000,
            slm_per_thread: 0,
            shader_local_mem: None,
            gpfifo_area,
            compute_gpfifo,
            dma_gpfifo,
            cmdq_buf,
            cmdq_wptr_bytes: 0,
            kernargs_buf,
            kernargs_wptr: 0,
            timeline_buf,
            timeline_value: 0,
            dma_staging_buf,
            unloaded: false,
        };

        gpu.setup_initial_gpfifos()?;
        let sm_lo = gpu.sm_version & 0xff;
        let sm_minor = if sm_lo > 0xf { sm_lo >> 4 } else { sm_lo };
        eprintln!(
            "[qwen-nv] Initialized {} (sm_{}{}, {} active SMs, {:.2} GB GDDR6X) in {:.2}s",
            gpu.chip_name,
            (gpu.sm_version >> 8) & 0xff,
            sm_minor,
            gpu.active_sms,
            gpu.mm.vram_size as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );
        Ok(gpu)
    }

    fn wait_for_gfw_boot(pci: &mut TinyGpuClient) -> Result<(), String> {
        let start = Instant::now();
        loop {
            let mask = pci.rreg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK)?;
            let prog = pci.rreg(NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0)?;
            if (mask & 1) == 1 && (prog & 0xff) == 0xff {
                return Ok(());
            }
            if start.elapsed() > Duration::from_secs(10) {
                return Err("Timeout waiting for GA102 GFW boot completion after PCI reset".into());
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn load_firmware_blobs() -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, [u32; 7]), String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let fw_dir = PathBuf::from(home).join(".cache/qwen-nv-rs/firmware");
        let gsp_img = fs::read(fw_dir.join("gsp_image.bin"))
            .map_err(|e| format!("Missing firmware {}/gsp_image.bin: {}", fw_dir.display(), e))?;
        let gsp_sig = fs::read(fw_dir.join("gsp_sig_ga10x.bin"))
            .map_err(|e| format!("Missing firmware {}/gsp_sig_ga10x.bin: {}", fw_dir.display(), e))?;
        let rm_booter = fs::read(fw_dir.join("rm_booter_image.bin"))
            .map_err(|e| format!("Missing firmware {}/rm_booter_image.bin: {}", fw_dir.display(), e))?;
        let sec2_booter = fs::read(fw_dir.join("sec2_booter_patched.bin"))
            .map_err(|e| format!("Missing firmware {}/sec2_booter_patched.bin: {}", fw_dir.display(), e))?;
        let meta_str = fs::read_to_string(fw_dir.join("fw_meta.json"))
            .map_err(|e| format!("Missing firmware {}/fw_meta.json: {}", fw_dir.display(), e))?;

        let parse_key = |key: &str| -> Result<u32, String> {
            let pat = format!("\"{}\":", key);
            let pos = meta_str.find(&pat).ok_or_else(|| format!("Key {} missing in fw_meta.json", key))?;
            let rest = meta_str[pos + pat.len()..].trim_start();
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse::<u32>().map_err(|e| e.to_string())
        };

        let meta = [
            parse_key("rm_monitor_code_off")?,
            parse_key("rm_monitor_data_off")?,
            parse_key("rm_manifest_off")?,
            parse_key("sec2_code_off")?,
            parse_key("sec2_code_sz")?,
            parse_key("sec2_data_off")?,
            parse_key("sec2_data_sz")?,
        ];
        Ok((gsp_img, gsp_sig, rm_booter, sec2_booter, meta))
    }

    fn parse_vbios_fwsec(pci: &mut TinyGpuClient, vram_size: u64) -> Result<(SysmemMapping, FwsecDesc, u64), String> {
        let mut vbios = vec![0u8; NV_PROM_SIZE];
        let chunk_sz = 64 * 1024;
        for off in (0..NV_PROM_SIZE).step_by(chunk_sz) {
            pci.bulk_read(0, (NV_PROM_BASE as u64) + (off as u64), &mut vbios[off..off + chunk_sz])?;
        }

        let mut vbios_off = 0usize;
        let mut block_size = 0usize;
        let expansion_rom_off: usize;
        loop {
            if vbios_off + 0x20 > vbios.len() {
                return Err("Failed to locate VBIOS_EXT image in GPU PROM".into());
            }
            let pci_blck = get_u16(&vbios, vbios_off + OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR) as usize;
            let imglen = (get_u16(&vbios, vbios_off + pci_blck + OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN) as usize)
                * PCI_ROM_IMAGE_BLOCK_SIZE;
            let code_type = vbios[vbios_off + pci_blck + OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE];
            if code_type == NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE {
                block_size = imglen;
            } else if code_type == NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT {
                expansion_rom_off = vbios_off - block_size;
                break;
            }
            vbios_off += imglen.max(512);
        }

        let bit_addr = 0x1b0usize;
        let bit_sig = get_u32(&vbios, bit_addr + 2);
        if bit_sig != 0x0054_4942 {
            return Err(format!("Invalid VBIOS BIT signature {:#x}", bit_sig));
        }
        let bit_hdr_sz = vbios[bit_addr + 8] as usize;
        let bit_tok_sz = vbios[bit_addr + 9] as usize;
        let bit_tok_cnt = vbios[bit_addr + 10] as usize;

        let mut ucode_desc_off = 0usize;
        let mut ucode_desc_size = 0usize;
        for i in 0..bit_tok_cnt {
            let toff = bit_addr + bit_hdr_sz + i * bit_tok_sz;
            let tok_id = vbios[toff];
            let data_ver = vbios[toff + 1];
            let data_sz = get_u16(&vbios, toff + 2);
            let data_ptr = (get_u32(&vbios, toff + 4) as usize) & 0xffff;
            if tok_id != BIT_TOKEN_FALCON_DATA || data_ver != 2 || data_sz < BIT_DATA_FALCON_DATA_V2_SIZE_4 {
                continue;
            }
            let ucode_tbl_ptr = expansion_rom_off + (get_u32(&vbios, data_ptr) as usize);
            let tbl_hdr_sz = vbios[ucode_tbl_ptr + 1] as usize;
            let tbl_ent_sz = vbios[ucode_tbl_ptr + 2] as usize;
            let tbl_ent_cnt = vbios[ucode_tbl_ptr + 3] as usize;
            for j in 0..tbl_ent_cnt {
                let eoff = ucode_tbl_ptr + tbl_hdr_sz + j * tbl_ent_sz;
                let app_id = vbios[eoff];
                if app_id == FALCON_UCODE_ENTRY_APPID_FWSEC_PROD {
                    let desc_ptr = get_u32(&vbios, eoff + 2) as usize;
                    ucode_desc_off = expansion_rom_off + desc_ptr;
                    let vdesc = get_u32(&vbios, ucode_desc_off);
                    ucode_desc_size = (vdesc >> 16) as usize;
                }
            }
        }
        if ucode_desc_off == 0 || ucode_desc_size < FALCON_UCODE_DESC_V3_SIZE_44 {
            return Err("FWSEC_PROD descriptor not found in VBIOS".into());
        }

        let desc = FwsecDesc {
            stored_size: get_u32(&vbios, ucode_desc_off + 4),
            pkc_data_off: get_u32(&vbios, ucode_desc_off + 8),
            interface_off: get_u32(&vbios, ucode_desc_off + 12),
            imem_pa: get_u32(&vbios, ucode_desc_off + 16),
            imem_sz: get_u32(&vbios, ucode_desc_off + 20),
            imem_va: get_u32(&vbios, ucode_desc_off + 24),
            dmem_pa: get_u32(&vbios, ucode_desc_off + 28),
            dmem_sz: get_u32(&vbios, ucode_desc_off + 32),
            eng_id_mask: get_u16(&vbios, ucode_desc_off + 36),
            ucode_id: vbios[ucode_desc_off + 38],
        };

        let sig_total = ucode_desc_size - FALCON_UCODE_DESC_V3_SIZE_44;
        let sig_slice = &vbios[ucode_desc_off + FALCON_UCODE_DESC_V3_SIZE_44 .. ucode_desc_off + FALCON_UCODE_DESC_V3_SIZE_44 + sig_total];
        let img_len = align_up(desc.stored_size as u64, 256) as usize;
        let mut patched = vbios[ucode_desc_off + ucode_desc_size .. ucode_desc_off + ucode_desc_size + img_len].to_vec();

        let frts_offset = vram_size - 0x20_0000;
        let mut frts_cmd = [0u8; 44];
        put_u32(&mut frts_cmd, 0, 1);   // readVbiosDesc.version
        put_u32(&mut frts_cmd, 4, 24);  // readVbiosDesc.size
        put_u32(&mut frts_cmd, 20, 2);  // readVbiosDesc.flags
        put_u32(&mut frts_cmd, 24, 1);  // frtsRegionDesc.version
        put_u32(&mut frts_cmd, 28, 20); // frtsRegionDesc.size
        put_u32(&mut frts_cmd, 32, (frts_offset >> 12) as u32);
        put_u32(&mut frts_cmd, 36, 0x100);
        put_u32(&mut frts_cmd, 40, 2);

        let app_hdr_off = (desc.imem_sz + desc.interface_off) as usize;
        let entry_cnt = patched[app_hdr_off + 3] as usize;
        let mut dmem_off = 0usize;
        for i in 0..entry_cnt {
            let eo = app_hdr_off + 4 + i * 8;
            if get_u32(&patched, eo) == FALCON_APPIF_ENTRY_ID_DMEMMAPPER {
                dmem_off = get_u32(&patched, eo + 4) as usize;
            }
        }
        let mapper_off = (desc.imem_sz as usize) + dmem_off;
        let cmd_in_buf_off = get_u32(&patched, mapper_off + 8) as usize;
        put_u32(&mut patched, mapper_off + 44, 0x15); // init_cmd = 0x15 (FRTS)

        let cmd_dst = (desc.imem_sz as usize) + cmd_in_buf_off;
        patched[cmd_dst..cmd_dst + 44].copy_from_slice(&frts_cmd);

        let sig_dst = (desc.imem_sz + desc.pkc_data_off) as usize;
        patched[sig_dst..sig_dst + 0x180].copy_from_slice(&sig_slice[sig_slice.len() - 0x180..]);

        let mut frts_mem = pci.alloc_sysmem(patched.len(), true)?;
        frts_mem.as_mut_slice()[..patched.len()].copy_from_slice(&patched);
        Ok((frts_mem, desc, frts_offset))
    }

    fn prepare_gsp_sw(pci: &mut TinyGpuClient, vram_size: u64) -> Result<GspState, String> {
        let (frts_img_mem, fwsec_desc, frts_offset) = Self::parse_vbios_fwsec(pci, vram_size)?;
        let (gsp_img, gsp_sig, rm_booter, sec2_booter, meta) = Self::load_firmware_blobs()?;

        // 1. GSP RPC Queues (`cmd_q` + `stat_q` = 2 * 256 KB + 4 KB page table)
        let queue_size = 0x40000usize;
        let queue_pte_cnt = (queue_size * 2) / 0x1000;
        let pte_cnt = queue_pte_cnt + ((queue_pte_cnt * 8 + 0xfff) / 0x1000);
        let pt_size = (pte_cnt * 8 + 0xfff) & !0xfff;
        let mut queues_mem = pci.alloc_sysmem(pt_size + queue_size * 2, false)?;
        for i in 0..queues_mem.paddrs_4k.len() {
            let pa = queues_mem.paddrs_4k[i];
            put_u64(queues_mem.as_mut_slice(), i * 8, pa);
        }

        let mut rm_args_mem = pci.alloc_sysmem(72, true)?;
        {
            let s = rm_args_mem.as_mut_slice();
            put_u64(s, 0, queues_mem.paddrs_4k[0]); // sharedMemPhysAddr
            put_u32(s, 8, pte_cnt as u32);          // pageTableEntryCount
            put_u64(s, 16, pt_size as u64);         // cmdQueueOffset
            put_u64(s, 24, (pt_size + queue_size) as u64); // statQueueOffset
            put_u8(s, 48, 1);                       // bDmemStack = 1
        }

        // Initialize `cmd_q` `msgqTxHeader` (32 bytes at `pt_size`)
        let msg_size = 0x1000usize;
        let msg_count = ((queue_size - 0x1000) / 0x1000) as u32;
        {
            let q = &mut queues_mem.as_mut_slice()[pt_size..pt_size + 32];
            put_u32(q, 0, 0);                 // version
            put_u32(q, 4, queue_size as u32); // size
            put_u32(q, 8, msg_size as u32);   // msgSize
            put_u32(q, 12, msg_count);        // msgCount = 63
            put_u32(q, 16, 0);                // writePtr = 0
            put_u32(q, 20, 1);                // flags = 1
            put_u32(q, 24, 32);               // rxHdrOff = 32
            put_u32(q, 28, 0x1000);           // entryOff = 0x1000
        }

        // 2. LIBOS Arguments (`LOGINIT`, `LOGINTR`, `LOGRM`, `LOGMNOC`, `LOGKRNL`, `RMARGS`)
        let logbuf_mem = pci.alloc_sysmem(2 << 20, true)?;
        let mut libos_args_mem = pci.alloc_sysmem(0x1000, true)?;
        let encode_id8 = |name: &str| -> u64 {
            let mut v = 0u64;
            for &b in name.as_bytes() {
                v = (v << 8) | (b as u64);
            }
            v
        };
        let log_names = ["LOGINIT", "LOGINTR", "LOGRM", "LOGMNOC", "LOGKRNL"];
        for (i, nm) in log_names.iter().enumerate() {
            let off = i * 32;
            let s = libos_args_mem.as_mut_slice();
            put_u64(s, off + 0, encode_id8(nm));
            put_u64(s, off + 8, logbuf_mem.paddrs_4k[0] + (i as u64) * 0x10000);
            put_u64(s, off + 16, 0x10000);
            put_u8(s, off + 24, LIBOS_MEMORY_REGION_CONTIGUOUS);
            put_u8(s, off + 25, LIBOS_MEMORY_REGION_LOC_SYSMEM);
        }
        {
            let off = 5 * 32;
            let s = libos_args_mem.as_mut_slice();
            put_u64(s, off + 0, encode_id8("RMARGS"));
            put_u64(s, off + 8, rm_args_mem.paddrs_4k[0]);
            put_u64(s, off + 16, 0x1000);
            put_u8(s, off + 24, LIBOS_MEMORY_REGION_CONTIGUOUS);
            put_u8(s, off + 25, LIBOS_MEMORY_REGION_LOC_SYSMEM);
        }

        // 3. Build 3-level Radix3 page table for `gsp_image.bin` (63.5 MB)
        // Allocated in <= 7.5 MB (`480 * 16KB`) `MAP_SYSMEM_FD` chunks so no single call can ever
        // exceed TinyGPU.app's hardcoded 511-entry `IOAddressSegment` scatter-gather cap!
        let mut npages = [0usize; 4];
        npages[3] = (gsp_img.len() + 0xfff) / 0x1000;
        for i in (1..=3).rev() {
            npages[i - 1] = ((npages[i] - 1) >> (LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1;
        }
        let offsets = [
            0usize,
            npages[0] * 0x1000,
            (npages[0] + npages[1]) * 0x1000,
            (npages[0] + npages[1] + npages[2]) * 0x1000,
        ];
        let total_radix_bytes = (offsets[3] + gsp_img.len() + 0xfff) & !0xfff;
        let max_sys_chunk = 4 << 20; // 4 MB chunks (strictly safe for DART IODMACommand segs[32]!)
        let mut gsp_radix3_chunks = Vec::<SysmemMapping>::new();
        let mut radix_paddrs_4k = Vec::<u64>::with_capacity(total_radix_bytes / 0x1000);
        let mut rem_radix = total_radix_bytes;
        while rem_radix > 0 {
            let take = rem_radix.min(max_sys_chunk);
            let ch = pci.alloc_sysmem(take, false)?;
            radix_paddrs_4k.extend_from_slice(&ch.paddrs_4k[..take / 0x1000]);
            gsp_radix3_chunks.push(ch);
            rem_radix -= take;
        }

        let write_radix = |chunks: &mut [SysmemMapping], mut dst_off: usize, mut src: &[u8]| {
            for ch in chunks.iter_mut() {
                let clen = ch.req_size;
                if dst_off >= clen {
                    dst_off -= clen;
                    continue;
                }
                let n = src.len().min(clen - dst_off);
                ch.as_mut_slice()[dst_off..dst_off + n].copy_from_slice(&src[..n]);
                src = &src[n..];
                dst_off = 0;
                if src.is_empty() {
                    break;
                }
            }
        };

        write_radix(&mut gsp_radix3_chunks, offsets[3], &gsp_img);
        for lvl in 0..3 {
            let cur_page_idx: usize = npages[..lvl + 1].iter().sum();
            let mut lvl_bytes = Vec::with_capacity(npages[lvl + 1] * 8);
            for k in 0..npages[lvl + 1] {
                lvl_bytes.extend_from_slice(&radix_paddrs_4k[cur_page_idx + k].to_le_bytes());
            }
            write_radix(&mut gsp_radix3_chunks, offsets[lvl], &lvl_bytes);
        }
        let radix_root_pa = radix_paddrs_4k[0];

        let mut gsp_sig_mem = pci.alloc_sysmem(gsp_sig.len(), true)?;
        gsp_sig_mem.as_mut_slice()[..gsp_sig.len()].copy_from_slice(&gsp_sig);

        let mut rm_booter_mem = pci.alloc_sysmem(rm_booter.len(), true)?;
        rm_booter_mem.as_mut_slice()[..rm_booter.len()].copy_from_slice(&rm_booter);

        let mut sec2_booter_mem = pci.alloc_sysmem(sec2_booter.len(), true)?;
        sec2_booter_mem.as_mut_slice()[..sec2_booter.len()].copy_from_slice(&sec2_booter);

        // 4. `GspFwWprMeta` (256 bytes)
        let vga_sz = 0x10_0000u64;
        let vga_off = vram_size - vga_sz;
        let frts_sz = 0x10_0000u64;
        let frts_off = vga_off - frts_sz;
        assert_eq!(frts_off, frts_offset, "FRTS offset mismatch");
        let boot_sz = rm_booter.len() as u64;
        let boot_off = frts_off - boot_sz;
        let radix3_sz = gsp_img.len() as u64;
        let gsp_off = align_down(boot_off - radix3_sz, 0x10000);
        let gsp_heap_sz = 0x810_0000u64;
        let gsp_heap_off = align_down(gsp_off - gsp_heap_sz, 0x10_0000);
        let wpr_st = align_down(gsp_heap_off - 0x1000, 0x10_0000);
        let non_wpr_sz = 0x10_0000u64;
        let non_wpr_off = align_down(wpr_st - non_wpr_sz, 0x10_0000);

        let mut wpr_meta_mem = pci.alloc_sysmem(256, true)?;
        {
            let m = wpr_meta_mem.as_mut_slice();
            put_u64(m, 0, GSP_FW_WPR_META_MAGIC);
            put_u64(m, 8, GSP_FW_WPR_META_REVISION);
            put_u64(m, 16, radix_root_pa);
            put_u64(m, 24, radix3_sz);
            put_u64(m, 32, rm_booter_mem.paddrs_4k[0]);
            put_u64(m, 40, boot_sz);
            put_u64(m, 48, meta[0] as u64); // monitorCodeOffset
            put_u64(m, 56, meta[1] as u64); // monitorDataOffset
            put_u64(m, 64, meta[2] as u64); // manifestOffset
            put_u64(m, 72, gsp_sig_mem.paddrs_4k[0]);
            put_u64(m, 80, 0x1000);
            put_u64(m, 88, non_wpr_off);
            put_u64(m, 96, non_wpr_off);
            put_u64(m, 104, non_wpr_sz);
            put_u64(m, 112, wpr_st);
            put_u64(m, 120, gsp_heap_off);
            put_u64(m, 128, gsp_heap_sz);
            put_u64(m, 136, gsp_off);
            put_u64(m, 144, boot_off);
            put_u64(m, 152, frts_off);
            put_u64(m, 160, frts_sz);
            put_u64(m, 168, vga_off);
            put_u64(m, 176, vram_size);
            put_u64(m, 184, vga_off);
            put_u64(m, 192, vga_sz);
        }

        let mut gsp = GspState {
            queues_mem,
            cmd_q_off: pt_size,
            stat_q_off: pt_size + queue_size,
            cmd_seq: 0,
            msg_size,
            msg_count,
            rm_args_mem,
            logbuf_mem,
            libos_args_mem,
            gsp_radix3_chunks,
            gsp_sig_mem,
            rm_booter_mem,
            wpr_meta_mem,
            frts_img_mem,
            sec2_booter_mem,
            fwsec_desc,
            sec2_code_off: meta[3],
            sec2_code_sz: meta[4],
            sec2_data_off: meta[5],
            sec2_data_sz: meta[6],
            next_handle: 0xcf00_0000,
            priv_root: 0xc1e0_0004,
            user_root: 0xc100_0000,
            user_device: 0,
            user_subdevice: 0,
            user_virtmem: 0,
            user_ctxshare: 0,
            method_sysmems: Vec::new(),
            grctx_sizes: Vec::new(),
            grctx_kept_bufs: Vec::new(),
        };

        // Pre-populate `cmd_q` with `GSP_SET_SYSTEM_INFO` and `SET_REGISTRY` before GSP starts!
        let mut sys_info = [0u8; 928];
        put_u64(&mut sys_info, 0, pci.bar_info(0)?.0);
        put_u64(&mut sys_info, 8, pci.bar_info(1)?.0);
        put_u64(&mut sys_info, 16, pci.bar_info(3)?.0);
        put_u64(&mut sys_info, 32, 0); // nvDomainBusDeviceFunc = 0 for remote/USB4
        put_u64(&mut sys_info, 72, 0x0000_7fff_ffff_f000);
        put_u32(&mut sys_info, 80, 0x88000);
        put_u32(&mut sys_info, 84, 0x1000);
        put_u32(&mut sys_info, 88, pci.read_config(PCI_VENDOR_ID, 4)? as u32);
        put_u32(&mut sys_info, 92, pci.read_config(PCI_SUBSYSTEM_VENDOR_ID, 4)? as u32);
        put_u32(&mut sys_info, 96, pci.read_config(PCI_REVISION_ID, 1)? as u32);
        put_u8(&mut sys_info, 840, 1); // bIsPassthru = 1
        Self::gsp_send_rpc(pci, &mut gsp, NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, &sys_info)?;

        // Registry table: `RMForcePcieConfigSave=1`, `RMSecBusResetEnable=1`
        let reg_entries = [("RMForcePcieConfigSave", 1u32), ("RMSecBusResetEnable", 1u32)];
        let hdr_sz = 8usize;
        let ents_sz = reg_entries.len() * 16;
        let mut ent_bytes = vec![0u8; ents_sz];
        let mut str_bytes = Vec::new();
        for (i, (k, v)) in reg_entries.iter().enumerate() {
            let eo = i * 16;
            put_u32(&mut ent_bytes, eo + 0, (hdr_sz + ents_sz + str_bytes.len()) as u32);
            put_u8(&mut ent_bytes, eo + 4, REGISTRY_TABLE_ENTRY_TYPE_DWORD);
            put_u32(&mut ent_bytes, eo + 8, *v);
            put_u32(&mut ent_bytes, eo + 12, 4);
            str_bytes.extend_from_slice(k.as_bytes());
            str_bytes.push(0);
        }
        let total_reg_sz = hdr_sz + ents_sz + str_bytes.len();
        let mut reg_blob = vec![0u8; hdr_sz];
        put_u32(&mut reg_blob, 0, total_reg_sz as u32);
        put_u32(&mut reg_blob, 4, reg_entries.len() as u32);
        reg_blob.extend_from_slice(&ent_bytes);
        reg_blob.extend_from_slice(&str_bytes);
        Self::gsp_send_rpc(pci, &mut gsp, NV_VGPU_MSG_FUNCTION_SET_REGISTRY, &reg_blob)?;

        Ok(gsp)
    }

    // -------------------------------------------------------------------------
    // Falcon & SEC2 Heavy-Secure Boot Helpers
    // -------------------------------------------------------------------------

    fn flcn_reset(pci: &mut TinyGpuClient, base: u32, riscv: bool) -> Result<(), String> {
        let eng_reg = if base == NV_PGSP_FALCON_BASE {
            NV_PGSP_FALCON_ENGINE
        } else {
            NV_PSEC_FALCON_ENGINE
        };
        pci.wreg(eng_reg, 1)?;
        thread::sleep(Duration::from_millis(100));
        pci.wreg(eng_reg, 0)?;

        let t0 = Instant::now();
        while ((pci.rreg(base + FLCN_HWCFG2)? >> 12) & 1) != 0 {
            if t0.elapsed() > Duration::from_secs(5) {
                return Err(format!("Falcon {:#x} mem_scrubbing timeout", base));
            }
            thread::sleep(Duration::from_millis(1));
        }

        if riscv {
            // core_select=1 (bit 4), valid=0 (bit 0), brfetch=1 (bit 8) -> 0x110
            pci.wreg(base + RISCV_BCR_CTRL, (1 << 4) | (1 << 8))?;
        } else if ((pci.rreg(base + FLCN_HWCFG2)? >> 10) & 1) == 1 {
            pci.wreg(base + RISCV_BCR_CTRL, 0)?;
            let t1 = Instant::now();
            while (pci.rreg(base + RISCV_BCR_CTRL)? & 1) == 0 {
                if t1.elapsed() > Duration::from_secs(5) {
                    return Err(format!("Falcon {:#x} BCR_CTRL valid timeout", base));
                }
                thread::sleep(Duration::from_millis(1));
            }
            let chip_id = pci.rreg(NV_PMC_BOOT_0)?;
            pci.wreg(base + FLCN_RM, chip_id)?;
        }
        Ok(())
    }

    fn flcn_disable_ctx_req(pci: &mut TinyGpuClient, base: u32) -> Result<(), String> {
        let ctl = pci.rreg(base + FLCN_FBIF_CTL)?;
        pci.wreg(base + FLCN_FBIF_CTL, ctl | (1 << 7))?; // allow_phys_no_ctx = 1
        pci.wreg(base + FLCN_DMACTL, 0)
    }

    fn flcn_start_cpu(pci: &mut TinyGpuClient, base: u32) -> Result<(), String> {
        let cpuctl = pci.rreg(base + FLCN_CPUCTL)?;
        if ((cpuctl >> 6) & 1) == 1 {
            pci.wreg(base + FLCN_CPUCTL_ALIAS, 0x2)
        } else {
            pci.wreg(base + FLCN_CPUCTL, 0x2) // startcpu = bit 1
        }
    }

    fn flcn_wait_halted(pci: &mut TinyGpuClient, base: u32) -> Result<(), String> {
        let t0 = Instant::now();
        while ((pci.rreg(base + FLCN_CPUCTL)? >> 4) & 1) == 0 {
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(format!("Falcon {:#x} did not halt within 10s", base));
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    fn flcn_execute_dma(
        pci: &mut TinyGpuClient,
        base: u32,
        cmd: u32,
        dest: u32,
        mem_off: u32,
        sysmem: u64,
        size: u32,
    ) -> Result<(), String> {
        while (pci.rreg(base + FLCN_DMATRFCMD)? & 1) != 0 {
            thread::sleep(Duration::from_micros(50));
        }
        let shifted = sysmem >> 8;
        pci.wreg(base + FLCN_DMATRFBASE, (shifted & 0xffff_ffff) as u32)?;
        pci.wreg(base + FLCN_DMATRFBASE1, ((shifted >> 32) & 0x1ff) as u32)?;

        let mut xfered = 0u32;
        while xfered < size {
            while (pci.rreg(base + FLCN_DMATRFCMD)? & 1) != 0 {
                thread::sleep(Duration::from_micros(50));
            }
            pci.wreg(base + FLCN_DMATRFMOFFS, dest.wrapping_add(xfered))?;
            pci.wreg(base + FLCN_DMATRFFBOFFS, mem_off.wrapping_add(xfered))?;
            pci.wreg(base + FLCN_DMATRFCMD, cmd)?;
            xfered += 256;
        }
        let t0 = Instant::now();
        while ((pci.rreg(base + FLCN_DMATRFCMD)? >> 1) & 1) == 0 {
            if t0.elapsed() > Duration::from_secs(5) {
                return Err(format!("Falcon {:#x} DMA idle timeout", base));
            }
            thread::sleep(Duration::from_micros(50));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn flcn_execute_hs(
        pci: &mut TinyGpuClient,
        base: u32,
        img_sysmem: u64,
        code_off: u32,
        data_off: u32,
        imem_pa: u32,
        imem_va: u32,
        imem_sz: u32,
        dmem_pa: u32,
        dmem_va: u32,
        dmem_sz: u32,
        pkc_off: u32,
        engid: u32,
        ucodeid: u32,
        mailbox: Option<u64>,
    ) -> Result<(u32, u32), String> {
        Self::flcn_disable_ctx_req(pci, base)?;
        // target=COHERENT_SYSMEM (1 at 1:0), mem_type=PHYSICAL (1 at 2:2) -> mask 0x7, val 0x5
        let tc0 = pci.rreg(base + FLCN_FBIF_TRANSCFG0)?;
        pci.wreg(base + FLCN_FBIF_TRANSCFG0, (tc0 & !0x7) | 0x5)?;

        // IMEM DMA: size=256B (6 << 8), imem=1 (1 << 4), sec=1 (1 << 2) -> 0x614
        let cmd_imem = (6 << 8) | (1 << 4) | (1 << 2);
        let imem_sys = img_sysmem.wrapping_add(code_off as u64).wrapping_sub(imem_va as u64);
        Self::flcn_execute_dma(pci, base, cmd_imem, imem_pa, imem_va, imem_sys, imem_sz)?;

        // DMEM DMA: size=256B (6 << 8), imem=0, sec=0 -> 0x600
        let cmd_dmem = 6 << 8;
        let dmem_sys = img_sysmem.wrapping_add(data_off as u64).wrapping_sub(dmem_va as u64);
        Self::flcn_execute_dma(pci, base, cmd_dmem, dmem_pa, dmem_va, dmem_sys, dmem_sz)?;

        pci.wreg(base + FLCN2_BROM_PARAADDR0, pkc_off)?;
        pci.wreg(base + FLCN2_ENGIDMASK, engid)?;
        pci.wreg(base + FLCN2_CURR_UCODE_ID, ucodeid)?;
        pci.wreg(base + FLCN2_MOD_SEL, 1)?; // RSA3K = 1
        pci.wreg(base + FLCN_BOOTVEC, imem_va)?;

        if let Some(mbx) = mailbox {
            pci.wreg(base + FLCN_MAILBOX0, (mbx & 0xffff_ffff) as u32)?;
            pci.wreg(base + FLCN_MAILBOX1, (mbx >> 32) as u32)?;
        }

        Self::flcn_start_cpu(pci, base)?;
        Self::flcn_wait_halted(pci, base)?;

        if mailbox.is_some() {
            Ok((pci.rreg(base + FLCN_MAILBOX0)?, pci.rreg(base + FLCN_MAILBOX1)?))
        } else {
            Ok((0, 0))
        }
    }

    fn boot_falcons_and_gsp(pci: &mut TinyGpuClient, gsp: &mut GspState) -> Result<(), String> {
        let d = gsp.fwsec_desc;
        Self::flcn_reset(pci, NV_PGSP_FALCON_BASE, false)?;
        Self::flcn_execute_hs(
            pci,
            NV_PGSP_FALCON_BASE,
            gsp.frts_img_mem.paddrs_4k[0],
            0,
            d.imem_sz,
            d.imem_pa,
            d.imem_va,
            d.imem_sz,
            d.dmem_pa,
            0,
            d.dmem_sz,
            d.pkc_data_off,
            d.eng_id_mask as u32,
            d.ucode_id as u32,
            None,
        )?;
        if pci.rreg(NV_PFB_PRI_MMU_WPR2_ADDR_HI)? == 0 {
            return Err("FWSEC FRTS failed to initialize WPR2".into());
        }

        Self::flcn_reset(pci, NV_PGSP_FALCON_BASE, true)?;
        let libos_pa = gsp.libos_args_mem.paddrs_4k[0];
        pci.wreg(NV_PGSP_FALCON_MAILBOX0, (libos_pa & 0xffff_ffff) as u32)?;
        pci.wreg(NV_PGSP_FALCON_MAILBOX1, (libos_pa >> 32) as u32)?;

        Self::flcn_reset(pci, NV_PSEC_FALCON_BASE, false)?;
        let (mbx0, mbx1) = Self::flcn_execute_hs(
            pci,
            NV_PSEC_FALCON_BASE,
            gsp.sec2_booter_mem.paddrs_4k[0],
            gsp.sec2_code_off,
            gsp.sec2_data_off,
            0,
            gsp.sec2_code_off,
            gsp.sec2_code_sz,
            0,
            0,
            gsp.sec2_data_sz,
            0x10,
            1,
            3,
            Some(gsp.wpr_meta_mem.paddrs_4k[0]),
        )?;
        if mbx0 != 0 {
            return Err(format!("SEC2 Booter failed: mailbox={:#010x}, {:#010x}", mbx0, mbx1));
        }

        pci.wreg(NV_PGSP_FALCON_BASE + FLCN_OS, 0)?;
        let riscv_ctl = pci.rreg(NV_PGSP_FALCON_BASE + RISCV_CPUCTL)?;
        if ((riscv_ctl >> 7) & 1) != 1 {
            return Err(format!("GSP RISC-V core not active (CPUCTL={:#x})", riscv_ctl));
        }

        // Wait for `stat_q` `entryOff == 0x1000` and `GSP_INIT_DONE` (`0x1001`)
        let t0 = Instant::now();
        loop {
            fence(Ordering::SeqCst);
            let entry_off = unsafe {
                ptr_read_u32_volatile(gsp.queues_mem.ptr.add(gsp.stat_q_off + 28))
            };
            if entry_off == 0x1000 {
                break;
            }
            if t0.elapsed() > Duration::from_secs(10) {
                return Err("Timeout waiting for GSP status queue header init".into());
            }
            thread::sleep(Duration::from_millis(2));
        }

        let _ = Self::gsp_wait_resp(pci, gsp, NV_VGPU_MSG_EVENT_GSP_INIT_DONE, Duration::from_secs(15))?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // GSP RPC Transport & CPU Sequencer
    // -------------------------------------------------------------------------

    fn gsp_checksum(data: &[u8]) -> u32 {
        let mut acc = 0u64;
        let chunks = data.len() / 8;
        for i in 0..chunks {
            acc ^= get_u64(data, i * 8);
        }
        let rem = data.len() % 8;
        if rem > 0 {
            let mut tail = [0u8; 8];
            tail[..rem].copy_from_slice(&data[chunks * 8..]);
            acc ^= u64::from_le_bytes(tail);
        }
        ((acc >> 32) as u32) ^ (acc as u32)
    }

    fn gsp_send_record(pci: &mut TinyGpuClient, gsp: &mut GspState, func: u32, payload: &[u8]) -> Result<(), String> {
        let rpc_len = (32 + payload.len()) as u32;
        let total_unpadded = 48 + 32 + payload.len();
        let elem_cnt = ((total_unpadded + gsp.msg_size - 1) / gsp.msg_size) as u32;
        let padded_len = (elem_cnt as usize) * gsp.msg_size;

        let mut pkt = vec![0u8; padded_len];
        // GSP_MSG_QUEUE_ELEMENT (48B)
        put_u32(&mut pkt, 36, gsp.cmd_seq);
        put_u32(&mut pkt, 40, elem_cnt);
        // rpc_message_header_v (32B at offset 48)
        put_u32(&mut pkt, 48 + 0, 3 << 24);
        put_u32(&mut pkt, 48 + 4, NV_VGPU_MSG_SIGNATURE_VALID);
        put_u32(&mut pkt, 48 + 8, rpc_len);
        put_u32(&mut pkt, 48 + 12, func);
        put_u32(&mut pkt, 48 + 16, NV_VGPU_MSG_RESULT_RPC_PENDING);
        put_u32(&mut pkt, 48 + 20, NV_VGPU_MSG_RESULT_RPC_PENDING);
        pkt[80..80 + payload.len()].copy_from_slice(payload);

        let csum = Self::gsp_checksum(&pkt[..total_unpadded]);
        put_u32(&mut pkt, 32, csum);

        let wp = unsafe { ptr_read_u32_volatile(gsp.queues_mem.ptr.add(gsp.cmd_q_off + 16)) } as usize;
        let ring_cap = (gsp.msg_count as usize) * gsp.msg_size;
        let ring_base = gsp.cmd_q_off + 0x1000;
        let write_off = wp * gsp.msg_size;
        let first = pkt.len().min(ring_cap - write_off);
        {
            let qslice = gsp.queues_mem.as_mut_slice();
            qslice[ring_base + write_off .. ring_base + write_off + first].copy_from_slice(&pkt[..first]);
            if first < pkt.len() {
                let rem = pkt.len() - first;
                qslice[ring_base .. ring_base + rem].copy_from_slice(&pkt[first..]);
            }
        }
        fence(Ordering::SeqCst);
        let new_wp = ((wp as u32) + elem_cnt) % gsp.msg_count;
        unsafe { ptr_write_u32_volatile(gsp.queues_mem.ptr.add(gsp.cmd_q_off + 16), new_wp) };
        fence(Ordering::SeqCst);

        gsp.cmd_seq += 1;
        pci.wreg(NV_PGSP_QUEUE_HEAD_0, 0)
    }

    fn gsp_send_rpc(pci: &mut TinyGpuClient, gsp: &mut GspState, func: u32, payload: &[u8]) -> Result<(), String> {
        let max_payload = gsp.msg_size * 16 - 48 - 32;
        let first_end = payload.len().min(max_payload);
        Self::gsp_send_record(pci, gsp, func, &payload[..first_end])?;
        let mut off = first_end;
        while off < payload.len() {
            let end = (off + max_payload).min(payload.len());
            Self::gsp_send_record(pci, gsp, NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD, &payload[off..end])?;
            off = end;
        }
        Ok(())
    }

    fn gsp_run_cpu_seq(pci: &mut TinyGpuClient, gsp: &GspState, seq_buf: &[u8]) -> Result<(), String> {
        if seq_buf.len() < 40 {
            return Err("Invalid CPU sequencer payload length".into());
        }
        let cmd_dwords = get_u32(seq_buf, 4) as usize;
        let mut idx = 0usize;
        let read_dw = |i: usize| -> u32 { get_u32(seq_buf, 40 + i * 4) };

        while idx < cmd_dwords {
            let op = read_dw(idx);
            idx += 1;
            match op {
                0x0 => {
                    let addr = read_dw(idx);
                    let val = read_dw(idx + 1);
                    idx += 2;
                    pci.wreg(addr, val)?;
                }
                0x1 => {
                    let addr = read_dw(idx);
                    let val = read_dw(idx + 1);
                    let mask = read_dw(idx + 2);
                    idx += 3;
                    let cur = pci.rreg(addr)?;
                    pci.wreg(addr, (cur & !mask) | (val & mask))?;
                }
                0x2 => {
                    let addr = read_dw(idx);
                    let mask = read_dw(idx + 1);
                    let val = read_dw(idx + 2);
                    idx += 5;
                    let t0 = Instant::now();
                    while (pci.rreg(addr)? & mask) != val {
                        if t0.elapsed() > Duration::from_secs(5) {
                            return Err(format!("CPU seq poll timeout on {:#x}", addr));
                        }
                        thread::sleep(Duration::from_micros(50));
                    }
                }
                0x3 => {
                    let us = read_dw(idx) as u64;
                    idx += 1;
                    thread::sleep(Duration::from_micros(us));
                }
                0x4 => {
                    let addr = read_dw(idx);
                    idx += 2;
                    let _ = pci.rreg(addr)?;
                }
                0x5 => {
                    Self::flcn_reset(pci, NV_PGSP_FALCON_BASE, false)?;
                    Self::flcn_disable_ctx_req(pci, NV_PGSP_FALCON_BASE)?;
                }
                0x6 => {
                    Self::flcn_start_cpu(pci, NV_PGSP_FALCON_BASE)?;
                }
                0x7 => {
                    Self::flcn_wait_halted(pci, NV_PGSP_FALCON_BASE)?;
                }
                0x8 => {
                    Self::flcn_reset(pci, NV_PGSP_FALCON_BASE, true)?;
                    let libos_pa = gsp.libos_args_mem.paddrs_4k[0];
                    pci.wreg(NV_PGSP_FALCON_MAILBOX0, (libos_pa & 0xffff_ffff) as u32)?;
                    pci.wreg(NV_PGSP_FALCON_MAILBOX1, (libos_pa >> 32) as u32)?;
                    Self::flcn_start_cpu(pci, NV_PSEC_FALCON_BASE)?;
                    let t0 = Instant::now();
                    while ((pci.rreg(NV_PGC6_BSI_SECURE_SCRATCH_14)? >> 26) & 1) == 0 {
                        if t0.elapsed() > Duration::from_secs(5) {
                            return Err("SEC2 boot_stage_3_handoff timeout".into());
                        }
                        thread::sleep(Duration::from_millis(1));
                    }
                    let mbx = pci.rreg(NV_PSEC_FALCON_BASE + FLCN_MAILBOX0)?;
                    if mbx != 0 {
                        return Err(format!("SEC2 resume failed: mailbox={:#x}", mbx));
                    }
                }
                other => return Err(format!("Unknown GSP CPU sequencer opcode {:#x}", other)),
            }
        }
        Ok(())
    }

    fn gsp_wait_resp(
        pci: &mut TinyGpuClient,
        gsp: &mut GspState,
        expected_func: u32,
        timeout: Duration,
    ) -> Result<Vec<u8>, String> {
        let t0 = Instant::now();
        let rx_ptr_addr = unsafe { gsp.queues_mem.ptr.add(gsp.cmd_q_off + 32) }; // cmd_q rxHdrOff = 32
        let stat_wp_addr = unsafe { gsp.queues_mem.ptr.add(gsp.stat_q_off + 16) };
        let stat_ring_base = gsp.stat_q_off + 0x1000;
        let ring_cap = (gsp.msg_count as usize) * gsp.msg_size;

        loop {
            fence(Ordering::SeqCst);
            let mut rp = unsafe { ptr_read_u32_volatile(rx_ptr_addr) };
            let wp = unsafe { ptr_read_u32_volatile(stat_wp_addr) };

            while rp != wp {
                let off = (rp as usize) * gsp.msg_size;
                let hdr_off = stat_ring_base + off + 0x30;
                let func = get_u32(gsp.queues_mem.as_slice(), hdr_off + 12);
                let rpc_len = get_u32(gsp.queues_mem.as_slice(), hdr_off + 8) as usize;
                let rpc_res = get_u32(gsp.queues_mem.as_slice(), hdr_off + 16);

                let payload_len = rpc_len.saturating_sub(32);
                let mut msg = vec![0u8; payload_len];
                let msg_ring_off = off + 0x50;
                if msg_ring_off + payload_len <= ring_cap {
                    msg.copy_from_slice(&gsp.queues_mem.as_slice()[stat_ring_base + msg_ring_off .. stat_ring_base + msg_ring_off + payload_len]);
                } else {
                    let first = ring_cap.saturating_sub(msg_ring_off);
                    msg[..first].copy_from_slice(&gsp.queues_mem.as_slice()[stat_ring_base + msg_ring_off .. stat_ring_base + ring_cap]);
                    msg[first..].copy_from_slice(&gsp.queues_mem.as_slice()[stat_ring_base .. stat_ring_base + (payload_len - first)]);
                }

                if func == NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER {
                    Self::gsp_run_cpu_seq(pci, gsp, &msg)?;
                } else if func == NV_VGPU_MSG_EVENT_OS_ERROR_LOG {
                    let log_txt = String::from_utf8_lossy(&msg[12.min(msg.len())..]);
                    eprintln!("[GSP ERROR LOG] {}", log_txt.trim_end_matches('\0'));
                }

                // Advance readPtr exactly matching `round_up(hdr.length, msgSize) // msgSize`
                let elems = ((rpc_len + gsp.msg_size - 1) / gsp.msg_size).max(1) as u32;
                rp = (rp + elems) % gsp.msg_count;
                unsafe { ptr_write_u32_volatile(rx_ptr_addr, rp) };
                fence(Ordering::SeqCst);

                if rpc_res != 0 {
                    return Err(format!("GSP RPC func={:#x} failed with status {:#x}", func, rpc_res));
                }
                if func == expected_func {
                    return Ok(msg);
                }
            }

            if t0.elapsed() > timeout {
                return Err(format!("Timeout waiting for GSP RPC response func={:#x}", expected_func));
            }
            thread::sleep(Duration::from_micros(50));
        }
    }

    // -------------------------------------------------------------------------
    // GSP Resource Manager Alloc / Control / Golden Context / GPFIFO Setup
    // -------------------------------------------------------------------------

    fn alloc_handle(gsp: &mut GspState) -> u32 {
        let h = gsp.next_handle;
        gsp.next_handle += 1;
        h
    }

    fn rpc_rm_control(
        pci: &mut TinyGpuClient,
        gsp: &mut GspState,
        h_object: u32,
        cmd: u32,
        params: &[u8],
        client: u32,
    ) -> Result<Vec<u8>, String> {
        let mut req = Vec::with_capacity(24 + params.len());
        req.resize(24, 0);
        put_u32(&mut req, 0, client);
        put_u32(&mut req, 4, h_object);
        put_u32(&mut req, 8, cmd);
        put_u32(&mut req, 16, params.len() as u32);
        req.extend_from_slice(params);

        if std::env::var("QWEN_NV_TRACE").is_ok() {
            let n = params.len().min(80);
            let hex: String = params[..n].iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("CTRL  client={:#x} obj={:#x} cmd={:#x} plen={} | {}", client, h_object, cmd, params.len(), hex);
        }
        Self::gsp_send_rpc(pci, gsp, NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, &req)?;
        let resp = Self::gsp_wait_resp(pci, gsp, NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, Duration::from_secs(10))?;
        Ok(resp[24..].to_vec())
    }

    fn rpc_set_page_directory(
        pci: &mut TinyGpuClient,
        gsp: &mut GspState,
        device: u32,
        h_vaspace: u32,
        pdir_paddr: u64,
        client: u32,
    ) -> Result<(), String> {
        let mut req = [0u8; 48];
        put_u32(&mut req, 0, client);
        put_u32(&mut req, 4, device);
        put_u32(&mut req, 8, 0xffff_ffff);
        put_u64(&mut req, 16, pdir_paddr);
        put_u32(&mut req, 24, MMU_CNT[0] as u32); // 4 entries in PDE3 on Turing/Ampere MMU v2!
        put_u32(&mut req, 28, 0x8);
        put_u32(&mut req, 32, h_vaspace);
        put_u32(&mut req, 36, 0);
        put_u32(&mut req, 40, 1);
        put_u32(&mut req, 44, 0xffff_ffff);
        Self::gsp_send_rpc(pci, gsp, NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, &req)?;
        let _ = Self::gsp_wait_resp(pci, gsp, NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, Duration::from_secs(10))?;
        Ok(())
    }

    fn promote_ctx(
        pci: &mut TinyGpuClient,
        mm: &mut VramMemoryManager,
        gsp: &mut GspState,
        client: u32,
        subdev: u32,
        ch_obj: u32,
        h_virtmem: u32,
        entries: &[(u16, u64, bool, bool)], // (buf_id, size, use_phys, use_virt)
        existing_bufs: Option<&[VramBuffer]>,
    ) -> Result<Vec<VramBuffer>, String> {
        let mut prom = vec![0u8; 560];
        put_u32(&mut prom, 0, 1); // engineType = 1
        put_u32(&mut prom, 12, client);
        put_u32(&mut prom, 16, ch_obj);
        put_u32(&mut prom, 20, h_virtmem);
        put_u32(&mut prom, 40, entries.len() as u32);

        let mut allocated = Vec::with_capacity(entries.len());
        for (i, &(buf_id, sz, use_p, use_v)) in entries.iter().enumerate() {
            let vbuf = if let Some(existing) = existing_bufs {
                VramBuffer {
                    va_addr: existing[i].va_addr,
                    size: existing[i].size,
                    paddrs: existing[i].paddrs.clone(),
                }
            } else {
                let pa = mm.palloc_bar1(pci, sz, 0x1000, true)?;
                let va = mm.alloc_vaddr(sz, 0x1000);
                mm.map_range(pci, va, &[(pa, align_up(sz, 0x1000))], false, false)?;
                VramBuffer {
                    va_addr: va,
                    size: align_up(sz, 0x1000),
                    paddrs: vec![(pa, align_up(sz, 0x1000))],
                }
            };

            let eo = 48 + i * 32;
            put_u64(&mut prom, eo + 0, if use_p { vbuf.paddrs[0].0 } else { 0 });
            put_u64(&mut prom, eo + 8, if use_v { vbuf.va_addr } else { 0 });
            put_u64(&mut prom, eo + 16, if use_p { sz } else { 0 });
            put_u32(&mut prom, eo + 24, if use_p { 4 } else { 0 });
            put_u16(&mut prom, eo + 28, buf_id);
            put_u8(&mut prom, eo + 30, u8::from(use_p));
            put_u8(&mut prom, eo + 31, u8::from(use_p && !use_v));
            allocated.push(vbuf);
        }

        let _ = Self::rpc_rm_control(pci, gsp, subdev, NV2080_CTRL_CMD_GPU_PROMOTE_CTX, &prom, client)?;
        Ok(allocated)
    }

    fn rpc_rm_alloc(
        pci: &mut TinyGpuClient,
        mm: &mut VramMemoryManager,
        gsp: &mut GspState,
        h_parent: u32,
        h_class: u32,
        params: &[u8],
        client: u32,
    ) -> Result<u32, String> {
        let mut patched_params = params.to_vec();
        if h_class == AMPERE_CHANNEL_GPFIFO_A {
            // Allocate 4KB contiguous BAR1 VRAM for `ramfcMem` & `instanceMem` + 20KB sysmem for `mthdbufMem`
            let ramfc_pa = mm.palloc_bar1(pci, 0x1000, 0x1000, true)?;
            let ramfc_va = mm.alloc_vaddr(0x1000, 0x1000);
            mm.map_range(pci, ramfc_va, &[(ramfc_pa, 0x1000)], false, false)?;
            gsp.grctx_kept_bufs.push(VramBuffer {
                va_addr: ramfc_va,
                size: 0x1000,
                paddrs: vec![(ramfc_pa, 0x1000)],
            });

            let mthd_mem = pci.alloc_sysmem(0x5000, true)?;
            let mthd_pa = mthd_mem.paddrs_4k[0];
            gsp.method_sysmems.push(mthd_mem);

            // instanceMem @ 144, ramfcMem @ 192, mthdbufMem @ 216
            put_u64(&mut patched_params, 144 + 0, ramfc_pa);
            put_u64(&mut patched_params, 144 + 8, 0x1000);
            put_u32(&mut patched_params, 144 + 16, 2);

            put_u64(&mut patched_params, 192 + 0, ramfc_pa);
            put_u64(&mut patched_params, 192 + 8, 0x200);
            put_u32(&mut patched_params, 192 + 16, 2);

            put_u64(&mut patched_params, 216 + 0, mthd_pa);
            put_u64(&mut patched_params, 216 + 8, 0x5000);
            put_u32(&mut patched_params, 216 + 16, 1);

            let h_obj_err = get_u32(&patched_params, 0);
            if client != gsp.priv_root && h_obj_err != 0 {
                let userd_mem0 = get_u32(&patched_params, 32) as u64;
                let userd_off0 = get_u64(&patched_params, 64);
                // userdMem @ 168
                put_u64(&mut patched_params, 168 + 0, userd_mem0 + userd_off0);
                put_u64(&mut patched_params, 168 + 8, 0x400);
                put_u32(&mut patched_params, 168 + 16, 2);
                // errorNotifierMem @ 248
                put_u64(&mut patched_params, 248 + 8, 0xecc);
            }
        }

        let obj = Self::alloc_handle(gsp);
        let mut req = Vec::with_capacity(32 + patched_params.len());
        req.resize(32, 0);
        put_u32(&mut req, 0, client);
        put_u32(&mut req, 4, h_parent);
        put_u32(&mut req, 8, obj);
        put_u32(&mut req, 12, h_class);
        put_u32(&mut req, 20, patched_params.len() as u32);
        req.extend_from_slice(&patched_params);

        if std::env::var("QWEN_NV_TRACE").is_ok() {
            let n = patched_params.len().min(64);
            let hex: String = patched_params[..n].iter().map(|b| format!("{:02x}", b)).collect();
            eprintln!("ALLOC client={:#x} parent={:#x} class={:#x} obj={:#x} plen={} | {}", client, h_parent, h_class, obj, patched_params.len(), hex);
        }
        Self::gsp_send_rpc(pci, gsp, NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, &req)?;
        let resp = Self::gsp_wait_resp(pci, gsp, NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, Duration::from_secs(10))?;
        let status = get_u32(&resp, 16);
        if status != 0 {
            return Err(format!("RM_ALLOC class={:#x} returned status={:#x}", h_class, status));
        }

        if h_class == AMPERE_COMPUTE_B && client != gsp.priv_root {
            let subdev = gsp.user_subdevice;
            let entries_phys: Vec<(u16, u64, bool, bool)> = gsp
                .grctx_sizes
                .iter()
                .filter(|&&(id, _, _, _, _)| id <= 2)
                .map(|&(id, sz, p, _, _)| (id, sz, p, false))
                .collect();
            let entries_virt: Vec<(u16, u64, bool, bool)> = gsp
                .grctx_sizes
                .iter()
                .filter(|&&(id, _, _, _, _)| id <= 2)
                .map(|&(id, sz, _, v, _)| (id, sz, false, v))
                .collect();

            let phys_bufs = Self::promote_ctx(pci, mm, gsp, client, subdev, h_parent, 0, &entries_phys, None)?;
            let _ = Self::promote_ctx(pci, mm, gsp, client, subdev, h_parent, 0, &entries_virt, Some(&phys_bufs))?;
            gsp.grctx_kept_bufs.extend(phys_bufs);
        }

        Ok(if h_class == NV01_ROOT { client } else { obj })
    }

    fn init_golden_image(
        pci: &mut TinyGpuClient,
        mm: &mut VramMemoryManager,
        gsp: &mut GspState,
    ) -> Result<(), String> {
        let priv_root = gsp.priv_root;
        let _ = Self::rpc_rm_alloc(pci, mm, gsp, 0, NV01_ROOT, &[0u8; 120], priv_root)?;

        let mut dev_p = [0u8; 56];
        put_u32(&mut dev_p, 4, priv_root);
        let dev = Self::rpc_rm_alloc(pci, mm, gsp, priv_root, NV01_DEVICE_0, &dev_p, priv_root)?;
        let subdev = Self::rpc_rm_alloc(pci, mm, gsp, dev, NV20_SUBDEVICE_0, &[0u8; 4], priv_root)?;
        let vaspace = Self::rpc_rm_alloc(pci, mm, gsp, dev, FERMI_VASPACE_A, &[0u8; 48], priv_root)?;

        // Reserve 512 MB for GSP Server Reserved PDEs
        let res_sz = 512u64 << 20;
        let res_va = mm.alloc_vaddr(res_sz, res_sz);
        let pt_levels = mm.reserve_pde_levels(res_va, res_sz);
        mm.flush_mmu(pci)?;

        let mut pde_params = [0u8; 184];
        put_u64(&mut pde_params, 8, res_sz);
        put_u64(&mut pde_params, 16, res_va);
        put_u64(&mut pde_params, 24, res_va + res_sz - 1);
        put_u32(&mut pde_params, 32, 3);
        for (i, &(lv, pt_pa)) in pt_levels.iter().enumerate() {
            let loff = 40 + i * 24;
            put_u64(&mut pde_params, loff + 0, pt_pa);
            put_u64(&mut pde_params, loff + 8, if i == 0 { (MMU_CNT[0] * 8) as u64 } else { 0x1000 });
            put_u32(&mut pde_params, loff + 16, 1); // aperture = video mem
            put_u8(&mut pde_params, loff + 20, MMU_COVERS[lv].trailing_zeros() as u8);
        }
        let _ = Self::rpc_rm_control(pci, gsp, vaspace, NV90F1_CTRL_CMD_COPY_SERVER_RESERVED_PDES, &pde_params, priv_root)?;
        // Sync back the PDE entries GSP-RM just wrote directly into VRAM so our host shadow cache preserves them!
        for &(_, pt_pa) in &pt_levels {
            mm.sync_pt_from_vram(pci, pt_pa)?;
        }

        // Golden GPFIFO channel
        let gold_pa = mm.palloc_bar1(pci, 0x1000, 0x1000, true)?;
        let gold_va = mm.alloc_vaddr(0x1000, 0x1000);
        mm.map_range(pci, gold_va, &[(gold_pa, 0x1000)], false, false)?;

        let mut gg_params = [0u8; 368];
        put_u64(&mut gg_params, 8, gold_va);
        put_u32(&mut gg_params, 16, 32);
        put_u32(&mut gg_params, 20, 0x200320); // flags
        put_u32(&mut gg_params, 28, vaspace);
        put_u64(&mut gg_params, 64, 0x100); // userdOffset[0] = 0x20 * 8
        put_u32(&mut gg_params, 128, 1); // engineType = 1
        put_u32(&mut gg_params, 132, 3); // cid = 3
        put_u64(&mut gg_params, 168 + 0, gold_pa + 0x100); // userdMem.base
        put_u64(&mut gg_params, 168 + 8, 0x20);
        put_u32(&mut gg_params, 168 + 16, 2);
        put_u32(&mut gg_params, 244, 0x1a); // internalFlags
        let ch_gpfifo = Self::rpc_rm_alloc(pci, mm, gsp, dev, AMPERE_CHANNEL_GPFIFO_A, &gg_params, priv_root)?;

        // Query `KGR_GET_CONTEXT_BUFFERS_INFO` (1664 bytes)
        let ctx_req = vec![0u8; 1664];
        let ctx_resp = Self::rpc_rm_control(pci, gsp, subdev, NV2080_CTRL_CMD_KGR_GET_CTX_BUFS_INFO, &ctx_req, priv_root)?;
        let ctx_info = |idx: usize, add: u64, align_override: Option<u64>| -> u64 {
            let sz = (get_u32(&ctx_resp, idx * 8) as u64) + add;
            let al = align_override.unwrap_or_else(|| (get_u32(&ctx_resp, idx * 8 + 4) as u64).max(0x1000));
            align_up(sz, al.max(1))
        };

        let gr_size = ctx_info(0, 0x40000, None);
        let patch_size = ctx_info(16, 0, None);
        let mut grctx_sizes = vec![
            (0u16, gr_size, true, true, false),
            (1u16, patch_size, true, true, true),
            (2u16, patch_size, true, true, false),
        ];
        for x in 3..=6usize {
            let sz = ctx_info(x + 14, 0, if x == 5 { Some(2 << 20) } else { None });
            grctx_sizes.push((x as u16, sz, false, true, false));
        }
        let sz9 = ctx_info(9 + 14, 0, None);
        let sz10 = ctx_info(10 + 14, 0, None);
        grctx_sizes.push((9, sz9, true, true, false));
        grctx_sizes.push((10, sz10, true, false, false));
        grctx_sizes.push((11, sz10, true, true, false));

        let non_local: Vec<(u16, u64, bool, bool)> = grctx_sizes
            .iter()
            .filter(|&&(_, _, _, _, loc)| !loc)
            .map(|&(id, sz, p, v, _)| (id, sz, p, v))
            .collect();
        let gold_bufs = Self::promote_ctx(pci, mm, gsp, priv_root, subdev, ch_gpfifo, 0, &non_local, None)?;
        gsp.grctx_kept_bufs.extend(gold_bufs);
        gsp.grctx_sizes = grctx_sizes;

        let _ = Self::rpc_rm_alloc(pci, mm, gsp, ch_gpfifo, AMPERE_COMPUTE_B, &[], priv_root)?;
        let _ = Self::rpc_rm_alloc(pci, mm, gsp, ch_gpfifo, AMPERE_DMA_COPY_B, &[], priv_root)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_gpfifo(
        pci: &mut TinyGpuClient,
        mm: &mut VramMemoryManager,
        gsp: &mut GspState,
        gpfifo_area: &VramBuffer,
        ctxshare: u32,
        channel_group: u32,
        offset: u64,
        entries: u32,
        compute: bool,
    ) -> Result<GpFifoRing, String> {
        let user_root = gsp.user_root;
        // Allocate 48 MB uncached notifier in high VRAM
        let notif_sz = 48u64 << 20;
        let notif_pa = mm.palloc_vram(notif_sz, 2 << 20)?;
        let notif_va = mm.alloc_vaddr(notif_sz, 2 << 20);
        mm.map_range(pci, notif_va, &[(notif_pa, notif_sz)], false, true)?;

        let area_pa = gpfifo_area.paddrs[0].0;
        let mut params = [0u8; 368];
        put_u32(&mut params, 0, notif_pa as u32); // hObjectError
        put_u32(&mut params, 4, area_pa as u32);  // hObjectBuffer
        put_u64(&mut params, 8, gpfifo_area.va_addr + offset);
        put_u32(&mut params, 16, entries);
        put_u32(&mut params, 24, ctxshare);
        put_u32(&mut params, 32, area_pa as u32); // hUserdMemory[0]
        put_u64(&mut params, 64, (entries as u64) * 8 + offset); // userdOffset[0]

        let gpfifo_h = Self::rpc_rm_alloc(pci, mm, gsp, channel_group, AMPERE_CHANNEL_GPFIFO_A, &params, user_root)?;
        if compute {
            let comp_obj = Self::rpc_rm_alloc(pci, mm, gsp, gpfifo_h, AMPERE_COMPUTE_B, &[], user_root)?;
            let mut dbg_params = [0u8; 12];
            put_u32(&mut dbg_params, 4, user_root);
            put_u32(&mut dbg_params, 8, comp_obj);
            let _ = Self::rpc_rm_alloc(pci, mm, gsp, gsp.user_device, GT200_DEBUGGER, &dbg_params, user_root)?;
        } else {
            let _ = Self::rpc_rm_alloc(pci, mm, gsp, gpfifo_h, AMPERE_DMA_COPY_B, &[], user_root)?;
        }

        let mut tok_req = [0u8; 4];
        put_u32(&mut tok_req, 0, 0xffff_ffff);
        let tok_resp = Self::rpc_rm_control(pci, gsp, gpfifo_h, NVC36F_CTRL_CMD_GET_WORK_SUBMIT_TOKEN, &tok_req, user_root)?;
        let token = get_u32(&tok_resp, 0);

        Ok(GpFifoRing {
            bar1_ring_off: area_pa + offset,
            bar1_gpput_off: area_pa + offset + (entries as u64) * 8 + (GPPUT_CTRL_OFFSET as u64),
            entries_count: entries,
            put_value: 0,
            token,
        })
    }

    // -------------------------------------------------------------------------
    // Memory Allocation & DMA Upload Helpers
    // -------------------------------------------------------------------------

    fn alloc_host_mapped_raw(
        pci: &mut TinyGpuClient,
        mm: &mut VramMemoryManager,
        size: usize,
    ) -> Result<HostMappedBuffer, String> {
        let sz_aligned = (size + 0xfff) & !0xfff;
        let mapping = pci.alloc_sysmem(sz_aligned, false)?;
        let va = mm.alloc_vaddr(sz_aligned as u64, 0x1000);
        let paddrs: Vec<(u64, u64)> = mapping.paddrs_4k.iter().map(|&p| (p, 0x1000u64)).collect();
        mm.map_range(pci, va, &paddrs, true, true)?;
        Ok(HostMappedBuffer {
            va_addr: va,
            size: sz_aligned,
            mapping,
        })
    }

    pub fn alloc_host_mapped(&mut self, size: usize) -> Result<HostMappedBuffer, String> {
        Self::alloc_host_mapped_raw(&mut self.pci, &mut self.mm, size)
    }

    pub fn alloc_vram(&mut self, size: u64) -> Result<VramBuffer, String> {
        let align = if size >= (8 << 20) { 2 << 20 } else { 0x1000 };
        let sz = align_up(size.max(0x1000), align);
        let pa = self.mm.palloc_vram(sz, align)?;
        let va = self.mm.alloc_vaddr(sz, align);
        self.mm.map_range(&mut self.pci, va, &[(pa, sz)], false, false)?;
        Ok(VramBuffer {
            va_addr: va,
            size: sz,
            paddrs: vec![(pa, sz)],
        })
    }

    pub fn upload_to_vram(&mut self, dst: &VramBuffer, offset: u64, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let chunk_cap = self.dma_staging_buf.size;
        let mut pos = 0usize;
        while pos < data.len() {
            let take = (data.len() - pos).min(chunk_cap);
            self.dma_staging_buf.as_mut_slice()[..take].copy_from_slice(&data[pos..pos + take]);
            fence(Ordering::SeqCst);

            let src_va = self.dma_staging_buf.va_addr;
            let dst_va = dst.va_addr + offset + (pos as u64);
            let sig_va = self.timeline_buf.va_addr;
            self.timeline_value += 1;
            let target_val = self.timeline_value;

            let mut cmds = Vec::with_capacity(16);
            // Copy methods on subchannel 4
            Self::push_nvm(&mut cmds, 4, NVC6B5_OFFSET_IN_UPPER, &[
                (src_va >> 32) as u32,
                (src_va & 0xffff_ffff) as u32,
                (dst_va >> 32) as u32,
                (dst_va & 0xffff_ffff) as u32,
            ]);
            Self::push_nvm(&mut cmds, 4, NVC6B5_LINE_LENGTH_IN, &[take as u32]);
            Self::push_nvm(&mut cmds, 4, NVC6B5_LAUNCH_DMA, &[NVC6B5_LAUNCH_DMA_COPY_FLAGS]);

            // Release 64-bit semaphore on `timeline_buf`? Note `NVC6B5_SET_SEMAPHORE_A` writes 32-bit payload in 4-word semaphore!
            // Instead, use GPFIFO host semaphore (`subchannel=0, NVC56F_SEM_ADDR_LO`) with `release_wfi=en, 64bit` OR `NVC6B5_SET_SEMAPHORE_A` + `NVC56F_SEM_ADDR_LO`!
            // Look at `ops_nv.py:208`: `NVC6B5_SET_SEMAPHORE_A` writes `value` (32-bit) at `value_addr` (clearing upper 32 bits in 4-word sem!).
            // Using `NVC6B5_SET_SEMAPHORE_A` when `target_val < u32::MAX` OR following `LAUNCH_DMA` (`flush_enable=1`) with `NVC56F_SEM_ADDR_LO` 64-bit release works cleanly for all 64-bit values!
            Self::push_nvm(&mut cmds, 4, NVC6B5_SET_SEMAPHORE_A, &[
                (sig_va >> 32) as u32,
                (sig_va & 0xffff_ffff) as u32,
                (target_val & 0xffff_ffff) as u32,
            ]);
            Self::push_nvm(&mut cmds, 4, NVC6B5_LAUNCH_DMA, &[NVC6B5_LAUNCH_DMA_SIG_FLAGS]);

            self.submit_gpfifo_cmds(false, &cmds)?;
            self.wait_timeline(target_val, Duration::from_secs(15))?;
            pos += take;
        }
        Ok(())
    }

    pub fn zero_vram(&mut self, dst: &VramBuffer) -> Result<(), String> {
        let z = vec![0u8; (dst.size as usize).min(self.dma_staging_buf.size)];
        let mut off = 0u64;
        while off < dst.size {
            let take = ((dst.size - off) as usize).min(z.len());
            self.upload_to_vram(dst, off, &z[..take])?;
            off += take as u64;
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // GPFIFO Pushbuffer Submission & Timeline Synchronization
    // -------------------------------------------------------------------------

    #[inline]
    pub fn push_nvm(cmds: &mut Vec<u32>, subch: u32, mthd: u32, args: &[u32]) {
        cmds.push((2u32 << 28) | ((args.len() as u32) << 16) | (subch << 13) | (mthd >> 2));
        cmds.extend_from_slice(args);
    }

    pub fn submit_gpfifo_cmds(&mut self, is_compute: bool, cmds: &[u32]) -> Result<(), String> {
        let byte_len = cmds.len() * 4;
        if self.cmdq_wptr_bytes + byte_len > self.cmdq_buf.size {
            self.cmdq_wptr_bytes = 0;
        }
        let off = self.cmdq_wptr_bytes;
        self.cmdq_wptr_bytes = (off + byte_len + 15) & !15;

        let dst_slice = &mut self.cmdq_buf.as_mut_slice()[off..off + byte_len];
        for (i, &w) in cmds.iter().enumerate() {
            dst_slice[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        fence(Ordering::SeqCst);

        let cmdq_va = self.cmdq_buf.va_addr + (off as u64);
        let gp_entry = ((cmdq_va / 4) << 2) | ((cmds.len() as u64) << 42) | (1u64 << 41);

        let ring = if is_compute {
            &mut self.compute_gpfifo
        } else {
            &mut self.dma_gpfifo
        };
        let slot = (ring.put_value % ring.entries_count) as u64;
        let next_put = (ring.put_value + 1) % ring.entries_count;
        let ring_off = ring.bar1_ring_off + slot * 8;
        let gpput_off = ring.bar1_gpput_off;
        let token = ring.token;
        ring.put_value = ring.put_value.wrapping_add(1);

        self.pci.bulk_write(1, ring_off, &gp_entry.to_le_bytes())?;
        self.pci.bulk_write(1, gpput_off, &next_put.to_le_bytes())?;
        // PCIe Spec §2.4.1 Producer-Consumer Flush: Non-Posted MRd on BAR1 flushes Apple Silicon WC gathering
        // buffers & Thunderbolt PCIe posted-write queues into VRAM BEFORE the BAR0 UC doorbell fires!
        let mut wc_flush = [0u8; 4];
        self.pci.bulk_read(1, gpput_off, &mut wc_flush)?;
        self.pci.ring_doorbell(token)?;
        if is_compute {
            // Non-Posted MRd on BAR0 guarantees the posted BAR0 doorbell MWr crosses Thunderbolt ASPM L0s/L1 immediately!
            let _ = self.pci.rreg(NV_PMC_BOOT_0)?;
        }
        Ok(())
    }

    pub fn poll_gsp_faults(&mut self) -> Result<(), String> {
        let rx_ptr_addr = unsafe { self.gsp.queues_mem.ptr.add(self.gsp.cmd_q_off + 32) };
        let stat_wp_addr = unsafe { self.gsp.queues_mem.ptr.add(self.gsp.stat_q_off + 16) };
        let stat_ring_base = self.gsp.stat_q_off + 0x1000;
        let ring_cap = (self.gsp.msg_count as usize) * self.gsp.msg_size;

        let mut rp = unsafe { ptr_read_u32_volatile(rx_ptr_addr) };
        let wp = unsafe { ptr_read_u32_volatile(stat_wp_addr) };
        if rp == wp {
            return Ok(());
        }

        let mut fault_err: Option<String> = None;
        while rp != wp {
            let off = (rp as usize) * self.gsp.msg_size;
            let hdr_off = stat_ring_base + off + 0x30;
            let func = get_u32(self.gsp.queues_mem.as_slice(), hdr_off + 12);
            let rpc_len = get_u32(self.gsp.queues_mem.as_slice(), hdr_off + 8) as usize;
            let payload_len = rpc_len.saturating_sub(32);
            let mut msg = vec![0u8; payload_len];
            let msg_ring_off = off + 0x50;
            if msg_ring_off + payload_len <= ring_cap {
                msg.copy_from_slice(&self.gsp.queues_mem.as_slice()[stat_ring_base + msg_ring_off .. stat_ring_base + msg_ring_off + payload_len]);
            }
            if func == NV_VGPU_MSG_EVENT_OS_ERROR_LOG {
                let txt = String::from_utf8_lossy(&msg[12.min(msg.len())..]).trim_end_matches('\0').to_string();
                eprintln!("[GSP ERROR LOG] {}", txt);
                fault_err = Some(format!("GSP OS_ERROR_LOG: {}", txt));
            } else if func == 4100 || func == 4101 {
                let hex: String = msg.iter().take(48).map(|b| format!("{:02x}", b)).collect();
                eprintln!("[GSP FAULT EVENT {}] {}", func, hex);
                fault_err = Some(format!("GSP GPU Fault Event {} ({})", func, hex));
            }
            let elems = ((rpc_len + self.gsp.msg_size - 1) / self.gsp.msg_size).max(1) as u32;
            rp = (rp + elems) % self.gsp.msg_count;
            unsafe { ptr_write_u32_volatile(rx_ptr_addr, rp) };
        }
        fence(Ordering::SeqCst);
        if let Some(e) = fault_err {
            Err(e)
        } else {
            Ok(())
        }
    }

    pub fn wait_timeline(&mut self, target: u64, timeout: Duration) -> Result<(), String> {
        let t0 = Instant::now();
        let sig_ptr = self.timeline_buf.as_ptr() as *const u64;
        let mut iter = 0u32;
        loop {
            let cur = unsafe { std::ptr::read_volatile(sig_ptr) };
            // Note: `NVC6B5_SET_SEMAPHORE_A` writes lower 32 bits (with upper 32 bits = 0 in word 1 of 4-word sem)
            if cur >= target || (target < (1u64 << 31) && (cur & 0xffff_ffff) >= target) {
                return Ok(());
            }
            iter = iter.wrapping_add(1);
            if (iter & 0x3fff) == 0 {
                self.poll_gsp_faults()?;
                if t0.elapsed() > Duration::from_millis(2) {
                    let _ = self.pci.ring_doorbell(self.compute_gpfifo.token);
                    let _ = self.pci.ring_doorbell(self.dma_gpfifo.token);
                }
            }
            if t0.elapsed() > timeout {
                self.poll_gsp_faults()?;
                return Err(format!(
                    "GPU timeline timeout waiting for {} (current={})",
                    target, cur
                ));
            }
            std::hint::spin_loop();
        }
    }

    fn setup_initial_gpfifos(&mut self) -> Result<(), String> {
        let sig_va = self.timeline_buf.va_addr;

        // 1. Setup Compute Queue (`AMPERE_COMPUTE_B` + `local_mem_window` + `shared_mem_window`)
        self.timeline_value += 1;
        let v1 = self.timeline_value;
        let mut c_cmds = Vec::new();
        Self::push_nvm(&mut c_cmds, 1, NVC6C0_SET_OBJECT, &[AMPERE_COMPUTE_B]);
        Self::push_nvm(&mut c_cmds, 1, NVC6C0_SET_LOCAL_MEM_WINDOW_A, &[
            (self.local_mem_window >> 32) as u32,
            (self.local_mem_window & 0xffff_ffff) as u32,
        ]);
        Self::push_nvm(&mut c_cmds, 1, NVC6C0_SET_SHARED_MEM_WINDOW_A, &[
            (self.shared_mem_window >> 32) as u32,
            (self.shared_mem_window & 0xffff_ffff) as u32,
        ]);
        Self::push_nvm(&mut c_cmds, 0, NVC56F_SEM_ADDR_LO, &[
            (sig_va & 0xffff_ffff) as u32,
            (sig_va >> 32) as u32,
            (v1 & 0xffff_ffff) as u32,
            (v1 >> 32) as u32,
            NVC56F_SEM_EXEC_REL_WFI_64_TS,
        ]);
        Self::push_nvm(&mut c_cmds, 0, NVC56F_NON_STALL_INTERRUPT, &[0]);
        self.submit_gpfifo_cmds(true, &c_cmds)?;
        self.wait_timeline(v1, Duration::from_secs(5))?;

        // 2. Setup Copy/DMA Queue (`AMPERE_DMA_COPY_B`)
        self.timeline_value += 1;
        let v2 = self.timeline_value;
        let mut d_cmds = Vec::new();
        Self::push_nvm(&mut d_cmds, 0, NVC56F_SEM_ADDR_LO, &[
            (sig_va & 0xffff_ffff) as u32,
            (sig_va >> 32) as u32,
            (v1 & 0xffff_ffff) as u32,
            (v1 >> 32) as u32,
            NVC56F_SEM_EXEC_ACQ_GEQ_64,
        ]);
        Self::push_nvm(&mut d_cmds, 4, NVC6C0_SET_OBJECT, &[AMPERE_DMA_COPY_B]);
        Self::push_nvm(&mut d_cmds, 0, NVC56F_SEM_ADDR_LO, &[
            (sig_va & 0xffff_ffff) as u32,
            (sig_va >> 32) as u32,
            (v2 & 0xffff_ffff) as u32,
            (v2 >> 32) as u32,
            NVC56F_SEM_EXEC_REL_WFI_64_TS,
        ]);
        Self::push_nvm(&mut d_cmds, 0, NVC56F_NON_STALL_INTERRUPT, &[0]);
        self.submit_gpfifo_cmds(false, &d_cmds)?;
        self.wait_timeline(v2, Duration::from_secs(5))?;
        Ok(())
    }

    /// Lightweight `< 25 us` Compute GPFIFO WFI+semaphore heartbeat so GA102 PGRAPH / Thunderbolt PCIe ASPM
    /// never deep-sleeps `compute_gpfifo` across multi-hour daemon idle periods.
    pub fn keepalive_compute_tick(&mut self) -> Result<(), String> {
        let sig_va = self.timeline_buf.va_addr;
        self.timeline_value += 1;
        let v = self.timeline_value;
        let mut c_cmds = Vec::with_capacity(8);
        Self::push_nvm(&mut c_cmds, 0, NVC56F_SEM_ADDR_LO, &[
            (sig_va & 0xffff_ffff) as u32,
            (sig_va >> 32) as u32,
            (v & 0xffff_ffff) as u32,
            (v >> 32) as u32,
            NVC56F_SEM_EXEC_REL_WFI_64_TS,
        ]);
        self.submit_gpfifo_cmds(true, &c_cmds)?;
        self.wait_timeline(v, Duration::from_secs(2))
    }

    pub fn ensure_local_memory(&mut self, required_bytes: u32) -> Result<(), String> {
        if self.slm_per_thread >= required_bytes {
            return Ok(());
        }
        let slm = align_up(required_bytes as u64, 32) as u32;
        let per_sm = align_up((slm as u64) * 32, 0x200) * (self.max_warps_per_sm as u64);
        let bytes_per_tpc = align_up(per_sm * (self.num_sm_per_tpc as u64), 0x8000);
        let total_slm = align_up(
            bytes_per_tpc * (self.num_tpc_per_gpc as u64) * (self.num_gpcs as u64),
            0x20000,
        );
        let slm_buf = self.alloc_vram(total_slm)?;
        let slm_va = slm_buf.va_addr;
        self.shader_local_mem = Some(slm_buf);
        self.slm_per_thread = slm;

        let sig_va = self.timeline_buf.va_addr;
        self.timeline_value += 1;
        let v = self.timeline_value;
        let mut cmds = Vec::new();
        Self::push_nvm(&mut cmds, 1, NVC6C0_SET_LOCAL_MEM_A, &[
            (slm_va >> 32) as u32,
            (slm_va & 0xffff_ffff) as u32,
        ]);
        Self::push_nvm(&mut cmds, 1, NVC6C0_SET_LOCAL_MEM_NON_THROTTLED_A, &[
            (bytes_per_tpc >> 32) as u32,
            (bytes_per_tpc & 0xffff_ffff) as u32,
            0xff,
        ]);
        Self::push_nvm(&mut cmds, 0, NVC56F_SEM_ADDR_LO, &[
            (sig_va & 0xffff_ffff) as u32,
            (sig_va >> 32) as u32,
            (v & 0xffff_ffff) as u32,
            (v >> 32) as u32,
            NVC56F_SEM_EXEC_REL_WFI_64_TS,
        ]);
        Self::push_nvm(&mut cmds, 0, NVC56F_NON_STALL_INTERRUPT, &[0]);
        self.submit_gpfifo_cmds(true, &cmds)?;
        self.wait_timeline(v, Duration::from_secs(5))
    }

    // -------------------------------------------------------------------------
    // Multi-Kernel `.cubin` ELF64 Loader & Chained QMDv3 Launcher
    // -------------------------------------------------------------------------

    pub fn load_cubin(&mut self, elf_bytes: &[u8]) -> Result<CubinModule, String> {
        if elf_bytes.len() < 64 || &elf_bytes[0..4] != b"\x7fELF" {
            return Err("Invalid ELF64 cubin header".into());
        }
        let e_shoff = get_u64(elf_bytes, 40) as usize;
        let e_shentsize = get_u16(elf_bytes, 58) as usize;
        let e_shnum = get_u16(elf_bytes, 60) as usize;
        let e_shstrndx = get_u16(elf_bytes, 62) as usize;

        #[derive(Clone)]
        struct Sec {
            name: String,
            sh_type: u32,
            sh_addr: u64,
            sh_offset: usize,
            sh_size: usize,
            sh_entsize: usize,
            sh_addralign: u64,
            img_off: u64,
        }

        let shstr_hdr = e_shoff + e_shstrndx * e_shentsize;
        let shstr_off = get_u64(elf_bytes, shstr_hdr + 24) as usize;
        let shstr_sz = get_u64(elf_bytes, shstr_hdr + 32) as usize;
        let shstr = &elf_bytes[shstr_off..shstr_off + shstr_sz];

        let read_cstr = |tab: &[u8], idx: usize| -> String {
            if idx >= tab.len() {
                return String::new();
            }
            let end = tab[idx..].iter().position(|&b| b == 0).map(|p| idx + p).unwrap_or(tab.len());
            String::from_utf8_lossy(&tab[idx..end]).into_owned()
        };

        let mut secs = Vec::with_capacity(e_shnum);
        for i in 0..e_shnum {
            let h = e_shoff + i * e_shentsize;
            let sh_name = get_u32(elf_bytes, h) as usize;
            let sh_type = get_u32(elf_bytes, h + 4);
            let sh_addr = get_u64(elf_bytes, h + 16);
            let sh_offset = get_u64(elf_bytes, h + 24) as usize;
            let sh_size = get_u64(elf_bytes, h + 32) as usize;
            let sh_addralign = get_u64(elf_bytes, h + 48);
            let sh_entsize = get_u64(elf_bytes, h + 56) as usize;
            secs.push(Sec {
                name: read_cstr(shstr, sh_name),
                sh_type,
                sh_addr,
                sh_offset,
                sh_size,
                sh_entsize,
                sh_addralign,
                img_off: sh_addr,
            });
        }

        // Lay out SHT_PROGBITS (type=1), skipping `.debug*`, aligned to >=256B (required by QMD cbuf & prefetch!)
        let mut image = Vec::<u8>::new();
        for s in &mut secs {
            if s.sh_type == 1 && !s.name.starts_with(".debug") {
                let align = s.sh_addralign.max(256) as usize;
                let pad = (align - (image.len() % align)) % align;
                image.resize(image.len() + pad, 0);
                s.img_off = image.len() as u64;
                image.extend_from_slice(&elf_bytes[s.sh_offset..s.sh_offset + s.sh_size]);
            }
        }

        // Parse `.symtab` + `.strtab` so `.nv.info` entries match exact kernel `sym_idx`!
        let mut sym_names = HashMap::<u32, String>::new();
        let mut sym_vals = Vec::<(u16, u64)>::new();
        if let (Some(sym_s), Some(str_s)) = (
            secs.iter().find(|s| s.name == ".symtab"),
            secs.iter().find(|s| s.name == ".strtab"),
        ) {
            let str_slice = &elf_bytes[str_s.sh_offset..str_s.sh_offset + str_s.sh_size];
            let esz = sym_s.sh_entsize.max(24);
            let cnt = sym_s.sh_size / esz;
            for idx in 0..cnt {
                let o = sym_s.sh_offset + idx * esz;
                let st_name = get_u32(elf_bytes, o) as usize;
                let st_shndx = get_u16(elf_bytes, o + 6);
                let st_val = get_u64(elf_bytes, o + 8);
                sym_names.insert(idx as u32, read_cstr(str_slice, st_name));
                sym_vals.push((st_shndx, st_val));
            }
        }

        // Allocate VRAM code buffer (+8KB prefetch guard)
        let code_buf = self.alloc_vram((image.len() as u64) + 0x2000)?;
        let base_va = code_buf.va_addr;

        // Apply non-debug SHT_REL (9) / SHT_RELA (4) CUDA relocations if any exist
        for s in &secs {
            if s.sh_type != 4 && s.sh_type != 9 {
                continue;
            }
            let tgt_name = if s.sh_type == 9 { &s.name[4..] } else { &s.name[5..] };
            if tgt_name == ".eh_frame" || tgt_name.starts_with(".debug") {
                continue;
            }
            let Some(tgt_sec) = secs.iter().find(|x| x.name == tgt_name) else { continue };
            let esz = s.sh_entsize.max(if s.sh_type == 9 { 16 } else { 24 });
            for i in 0..(s.sh_size / esz) {
                let ro = s.sh_offset + i * esz;
                let r_off = get_u64(elf_bytes, ro);
                let r_info = get_u64(elf_bytes, ro + 8);
                let r_add = if s.sh_type == 4 { get_u64(elf_bytes, ro + 16) } else { 0 };
                let sym_i = (r_info >> 32) as usize;
                let r_type = (r_info & 0xffff_ffff) as u32;
                let (shndx, sval) = sym_vals[sym_i];
                let sym_va = base_va + secs[shndx as usize].img_off + sval + r_add;
                let apply_off = (tgt_sec.img_off + r_off) as usize;
                match r_type {
                    2 => put_u64(&mut image, apply_off, sym_va),
                    0x38 => put_u32(&mut image, apply_off + 4, (sym_va & 0xffff_ffff) as u32),
                    0x39 => put_u32(&mut image, apply_off + 4, (sym_va >> 32) as u32),
                    other => return Err(format!("Unsupported CUDA ELF reloc type {:#x}", other)),
                }
            }
        }

        // Upload image to VRAM
        self.upload_to_vram(&code_buf, 0, &image)?;

        // Parse global `.nv.info` per-symbol `regs_usage` (0x2f) & `lcmem_usage` (0x12)
        let mut regs_by_name = HashMap::<String, u32>::new();
        let mut lcmem_by_name = HashMap::<String, u32>::new();
        if let Some(nvi) = secs.iter().find(|s| s.name == ".nv.info") {
            let blob = &elf_bytes[nvi.sh_offset..nvi.sh_offset + nvi.sh_size];
            let mut off = 0usize;
            while off + 4 <= blob.len() {
                let fmt = blob[off];
                let attr = blob[off + 1];
                let sz = get_u16(blob, off + 2) as usize;
                if fmt == 4 && sz == 8 && off + 12 <= blob.len() {
                    let sym_idx = get_u32(blob, off + 4);
                    let val = get_u32(blob, off + 8);
                    if let Some(kname) = sym_names.get(&sym_idx) {
                        if attr == 0x2f {
                            regs_by_name.insert(kname.clone(), val);
                        } else if attr == 0x12 {
                            lcmem_by_name.insert(kname.clone(), val + 0x240);
                        }
                    }
                }
                off += if fmt == 4 { 4 + sz } else { 4 };
            }
        }

        // Build each `GpuKernel` with strictly per-kernel constant banks & `.nv.info.<kname>`!
        let mut max_lcmem = 0x240u32;
        let mut kernels = HashMap::new();
        for s in &secs {
            let Some(kname) = s.name.strip_prefix(".text.") else { continue };
            let prog_addr = base_va + s.img_off;
            let prog_sz = s.sh_size;
            let regs_usage = *regs_by_name.get(kname).unwrap_or(&32);
            let lcmem_usage = *lcmem_by_name.get(kname).unwrap_or(&0x240);
            max_lcmem = max_lcmem.max(lcmem_usage);

            let shmem_raw = secs
                .iter()
                .find(|x| x.name == format!(".nv.shared.{}", kname))
                .map(|x| x.sh_size as u32)
                .unwrap_or(0);
            let shmem_usage = align_up((0x400 + shmem_raw) as u64, 128) as u32;

            // Parse `.nv.info.<kname>` for `EIATTR_PARAM_CBANK` (0x0a)
            let mut param_end = 0x160u32;
            if let Some(knvi) = secs.iter().find(|x| x.name == format!(".nv.info.{}", kname)) {
                let blob = &elf_bytes[knvi.sh_offset..knvi.sh_offset + knvi.sh_size];
                let mut off = 0usize;
                while off + 4 <= blob.len() {
                    let fmt = blob[off];
                    let attr = blob[off + 1];
                    let sz = get_u16(blob, off + 2) as usize;
                    if fmt == 4 && attr == 0x0a && sz >= 8 && off + 12 <= blob.len() {
                        let cbank_off = get_u16(blob, off + 8) as u32;
                        let cbank_sz = get_u16(blob, off + 10) as u32;
                        param_end = param_end.max(cbank_off + cbank_sz);
                    }
                    off += if fmt == 4 { 4 + sz } else { 4 };
                }
            }

            // Collect `.nv.constant<idx>.<kname>` AND any module-wide `.nv.constant<idx>`
            let mut cbuf0_size = param_end.max(0x180);
            let mut extra_cbufs = Vec::new();
            let suffix = format!(".{}", kname);
            for cs in &secs {
                if let Some(rest) = cs.name.strip_prefix(".nv.constant") {
                    let idx_str_opt = if let Some(idx_s) = rest.strip_suffix(&suffix) {
                        if !idx_s.contains('.') { Some(idx_s) } else { None }
                    } else if !rest.contains('.') {
                        Some(rest)
                    } else {
                        None
                    };
                    if let Some(idx_s) = idx_str_opt {
                        if let Ok(cb_idx) = idx_s.parse::<usize>() {
                            if cb_idx == 0 {
                                cbuf0_size = cbuf0_size.max(cs.sh_size as u32);
                            } else {
                                let aligned_cb_sz = align_up(cs.sh_size as u64, 16) as u32;
                                extra_cbufs.push((cb_idx, base_va + cs.img_off, aligned_cb_sz));
                            }
                        }
                    }
                }
            }
            cbuf0_size = align_up(cbuf0_size as u64, 16) as u32;

            // Populate Ampere `cbuf0[6..12]` driver constants (`shared_mem_window`, `local_mem_window`, `0xfffdc0`)
            let mut cbuf0_template = vec![0u8; cbuf0_size as usize];
            put_u64(&mut cbuf0_template, 24, self.shared_mem_window);
            put_u64(&mut cbuf0_template, 32, self.local_mem_window);
            put_u64(&mut cbuf0_template, 40, 0xfffdc0);

            kernels.insert(
                kname.to_string(),
                GpuKernel {
                    name: kname.to_string(),
                    prog_addr,
                    prog_sz,
                    regs_usage,
                    shmem_usage,
                    lcmem_usage,
                    cbuf0_size,
                    cbuf0_template,
                    extra_cbufs,
                    qmd_template: QmdV3::new_zeroed(),
                },
            );
        }

        // Allocate Shader Local Memory (`0x240` driver frame) BEFORE finalizing QMD `slm_per_thread`!
        self.ensure_local_memory(max_lcmem)?;
        let slm_pt = self.slm_per_thread;
        let sass_ver = self.sass_version;

        for kern in kernels.values_mut() {
            kern.qmd_template.init_ampere_template(
                kern.prog_addr,
                kern.prog_sz,
                kern.shmem_usage,
                kern.regs_usage,
                slm_pt,
                sass_ver,
                0,
                kern.cbuf0_size,
            );
            for &(cb_idx, cb_addr, cb_sz) in &kern.extra_cbufs {
                kern.qmd_template.set_cbuf(cb_idx, cb_addr, cb_sz);
            }
        }

        Ok(CubinModule { code_buf, kernels })
    }

    /// Stage `(cbuf0 + params)` and `QmdV3` into the 4 MB host-mapped `kernargs_buf` ring and return `qmd_va`.
    pub fn prepare_qmd(
        &mut self,
        kern: &GpuKernel,
        grid: (u32, u32, u32),
        block: (u16, u16, u16),
        arg_bytes: &[u8],
        release_sig: Option<(u64, u64)>,
    ) -> (u64, usize) {
        let cbuf_aligned = ((kern.cbuf0_size as usize) + 255) & !255;
        let slot_size = cbuf_aligned + 256;
        if self.kernargs_wptr + slot_size > self.kernargs_buf.size {
            self.kernargs_wptr = 0;
        }
        let cbuf_off = self.kernargs_wptr;
        let qmd_off = cbuf_off + cbuf_aligned;
        self.kernargs_wptr = (qmd_off + 256 + 255) & !255;

        let cbuf_va = self.kernargs_buf.va_addr + (cbuf_off as u64);
        let qmd_va = self.kernargs_buf.va_addr + (qmd_off as u64);

        let mut qmd = kern.qmd_template.clone();
        qmd.set_cbuf(0, cbuf_va, kern.cbuf0_size);
        qmd.set_launch_dims(grid, block);
        if let Some((sig_va, val)) = release_sig {
            qmd.set_release0(sig_va, val);
        }

        let buf = self.kernargs_buf.as_mut_slice();
        buf[cbuf_off .. cbuf_off + kern.cbuf0_template.len()].copy_from_slice(&kern.cbuf0_template);
        // Populate CUDA Ampere `c[0x0][0x00..0x18]` launch dimensions (`blockDim.{x,y,z}` + `gridDim.{x,y,z}`)!
        put_u32(buf, cbuf_off + 0x00, block.0 as u32);
        put_u32(buf, cbuf_off + 0x04, block.1 as u32);
        put_u32(buf, cbuf_off + 0x08, block.2 as u32);
        put_u32(buf, cbuf_off + 0x0c, grid.0);
        put_u32(buf, cbuf_off + 0x10, grid.1);
        put_u32(buf, cbuf_off + 0x14, grid.2);
        assert!(0x160 + arg_bytes.len() <= kern.cbuf0_template.len(), "Kernel args exceed cbuf0_size");
        buf[cbuf_off + 0x160 .. cbuf_off + 0x160 + arg_bytes.len()].copy_from_slice(arg_bytes);
        buf[qmd_off .. qmd_off + 256].copy_from_slice(&qmd.bytes);

        (qmd_va, qmd_off)
    }

    pub fn link_dependent_qmd(&mut self, prev_qmd_off: usize, next_qmd_va: u64) {
        let buf = self.kernargs_buf.as_mut_slice();
        let mut qmd = QmdV3 {
            bytes: buf[prev_qmd_off..prev_qmd_off + 256].try_into().unwrap(),
        };
        qmd.set_dependent_qmd0(next_qmd_va);
        buf[prev_qmd_off..prev_qmd_off + 256].copy_from_slice(&qmd.bytes);
    }

    #[inline]
    pub fn kernargs_buf_mut_slice_at(&mut self, off: usize) -> &mut [u8] {
        &mut self.kernargs_buf.as_mut_slice()[off..]
    }

    /// Launch the head QMD of a single kernel or hardware-chained QMD sequence (`1` doorbell total!).
    pub fn launch_qmd_chain(&mut self, head_qmd_va: u64, invalidate_caches: bool) -> Result<(), String> {
        fence(Ordering::SeqCst);
        let mut cmds = Vec::with_capacity(8);
        if invalidate_caches {
            Self::push_nvm(&mut cmds, 1, NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, &[NVC6C0_INVALIDATE_FLAGS]);
        }
        Self::push_nvm(&mut cmds, 1, NVC6C0_SEND_PCAS_A, &[(head_qmd_va >> 8) as u32]);
        Self::push_nvm(&mut cmds, 1, NVC6C0_SEND_SIGNALING_PCAS2_B, &[9]);
        self.submit_gpfifo_cmds(true, &cmds)
    }

    fn emergency_halt_falcons(pci: &mut TinyGpuClient) {
        let _ = pci.wreg(NV_PGSP_FALCON_ENGINE, 1);
        let _ = pci.wreg(NV_PSEC_FALCON_ENGINE, 1);
        thread::sleep(Duration::from_millis(25));
    }

    pub fn shutdown_clean(&mut self) {
        if self.unloaded {
            return;
        }
        self.unloaded = true;
        // Passively drain any in-flight GPU work for up to 1.5s WITHOUT re-ringing compute_gpfifo
        // (never wake an unstarted QMD milliseconds before unloading GSP-RM!).
        let sig_ptr = self.timeline_buf.as_ptr() as *const u64;
        let t_drain = Instant::now();
        while t_drain.elapsed() < Duration::from_millis(1500) {
            let cur = unsafe { std::ptr::read_volatile(sig_ptr) };
            let tgt = self.timeline_value;
            if cur >= tgt || (tgt < (1u64 << 31) && (cur & 0xffff_ffff) >= tgt) {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let mut unload_req = [0u8; 8];
        put_u32(&mut unload_req, 4, 1 << 6); // __GPU_STATE_FLAGS_FAST_UNLOAD
        let ok = Self::gsp_send_rpc(
            &mut self.pci,
            &mut self.gsp,
            NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER,
            &unload_req,
        )
        .and_then(|_| {
            Self::gsp_wait_resp(
                &mut self.pci,
                &mut self.gsp,
                NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER,
                Duration::from_secs(3),
            )
        })
        .is_ok();
        if !ok {
            Self::emergency_halt_falcons(&mut self.pci);
        }
        thread::sleep(Duration::from_millis(15));
    }
}

impl Drop for NvGpu {
    fn drop(&mut self) {
        self.shutdown_clean();
    }
}

#[inline]
unsafe fn ptr_read_u32_volatile(p: *const u8) -> u32 {
    std::ptr::read_volatile(p as *const u32)
}

#[inline]
unsafe fn ptr_write_u32_volatile(p: *mut u8, val: u32) {
    std::ptr::write_volatile(p as *mut u32, val);
}
