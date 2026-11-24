//! GA102 (`sm_86`) User-Space Hardware Driver, ELF64 `.cubin` Loader, and QMDv3 Launcher.
#![allow(dead_code)]

pub mod defs;
mod gsp;
mod mmu;
mod tinygpu;

use std::collections::HashMap;
use std::sync::atomic::{fence, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub use defs::{get_u16, get_u32, get_u64, put_u32, put_u64};
pub use mmu::{align_up, HostMappedBuffer, VramBuffer, VramCheckpoint};

use defs::*;
use gsp::{GpFifoRing, GspState};
use mmu::VramMemoryManager;
use tinygpu::TinyGpuClient;

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
    _code_buf: VramBuffer,
    kernels: HashMap<String, GpuKernel>,
}

impl CubinModule {
    #[inline]
    pub fn kernel(&self, name: &str) -> Result<&GpuKernel, String> {
        self.kernels
            .get(name)
            .ok_or_else(|| format!("Missing CUDA kernel '{}' in loaded .cubin", name))
    }

    #[inline]
    pub fn try_kernel(&self, name: &str) -> Option<&GpuKernel> {
        self.kernels.get(name)
    }
}

// -----------------------------------------------------------------------------
// Top-Level GA102 Hardware Context (`NvGpu`)
// -----------------------------------------------------------------------------

pub struct NvGpu {
    pci: TinyGpuClient,
    mm: VramMemoryManager,
    gsp: GspState,
    chip_id: u32,
    chip_name: String,
    sm_version: u32,
    sass_version: u32,
    pub active_sms: u32,
    num_gpcs: u32,
    num_tpc_per_gpc: u32,
    num_sm_per_tpc: u32,
    max_warps_per_sm: u32,
    shared_mem_window: u64,
    local_mem_window: u64,
    slm_per_thread: u32,
    shader_local_mem: Option<VramBuffer>,

    gpfifo_area: VramBuffer,
    compute_gpfifo: GpFifoRing,
    dma_gpfifo: GpFifoRing,

    cmdq_buf: HostMappedBuffer,
    cmdq_wptr_bytes: usize,
    kernargs_buf: HostMappedBuffer,
    kernargs_wptr: usize,

    timeline_buf: HostMappedBuffer,
    timeline_value: u64,
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
            GspState::emergency_halt_falcons(&mut pci);
            let cmd_pre = pci.read_config(PCI_COMMAND, 2).unwrap_or(0);
            let _ = pci.write_config(PCI_COMMAND, cmd_pre & !PCI_COMMAND_MASTER, 2);
            thread::sleep(Duration::from_millis(25));
            pci.reset_pci()?;
            thread::sleep(Duration::from_millis(650));
            pci.remap_bars()?;
            let pci_cmd2 = pci.read_config(PCI_COMMAND, 2)?;
            pci.write_config(PCI_COMMAND, (pci_cmd2 | 0x2) & !PCI_COMMAND_MASTER, 2)?;
            GspState::wait_for_gfw_boot(&mut pci)?;
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
        let mut gsp = GspState::prepare_gsp_sw(&mut pci, vram_size)?;

        // Boot GSP-RM via Falcon (FRTS) + SEC2 (Booter)
        if let Err(e) = GspState::boot_falcons_and_gsp(&mut pci, &mut gsp) {
            GspState::emergency_halt_falcons(&mut pci);
            return Err(e);
        }

        // From this point on, GSP-RM holds DART IOVAs: if ANY setup step fails, we MUST unload GSP
        // and halt Falcon/RISC-V DMA BEFORE `pci` drops and `TinyGPU` calls `IODMACommand::CompleteDMA`!
        let init_after_gsp = (|| -> Result<_, String> {
            // Clear BAR1 block FIRST so BAR1 accesses physical VRAM directly before flushing root PDE!
            pci.wreg(NV_PBUS_BAR1_BLOCK, 0)?;
            mm.flush_mmu(&mut pci)?;

            // Initialize GSP Golden Context Image + User VASpace / Channels / SM Topology
            GspState::init_golden_image(&mut pci, &mut mm, &mut gsp)?;
            let topo = gsp.init_user_channels_and_topology(&mut pci, &mut mm)?;

            let cmdq_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 2 << 20)?;
            let kernargs_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 4 << 20)?;
            let timeline_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 0x1000)?;
            let dma_staging_buf = Self::alloc_host_mapped_raw(&mut pci, &mut mm, 4 << 20)?;

            Ok((topo, cmdq_buf, kernargs_buf, timeline_buf, dma_staging_buf))
        })();

        let (topo, cmdq_buf, kernargs_buf, timeline_buf, dma_staging_buf) = match init_after_gsp {
            Ok(tup) => tup,
            Err(e) => {
                gsp.unload_guest_driver(&mut pci, Duration::from_secs(2));
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
            sm_version: topo.sm_version,
            sass_version: topo.sass_version,
            active_sms: topo.active_sms,
            num_gpcs: topo.num_gpcs,
            num_tpc_per_gpc: topo.num_tpc_per_gpc,
            num_sm_per_tpc: topo.num_sm_per_tpc,
            max_warps_per_sm: topo.max_warps_per_sm,
            shared_mem_window: 0x7294_0000_0000,
            local_mem_window: 0x7293_0000_0000,
            slm_per_thread: 0,
            shader_local_mem: None,
            gpfifo_area: topo.gpfifo_area,
            compute_gpfifo: topo.compute_gpfifo,
            dma_gpfifo: topo.dma_gpfifo,
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
        Ok(HostMappedBuffer::new(va, sz_aligned, mapping))
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

        Ok(CubinModule { _code_buf: code_buf, kernels })
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



    #[inline]
    pub fn vram_size(&self) -> u64 {
        self.mm.vram_size
    }

    #[inline]
    pub fn free_high_vram(&self) -> u64 {
        self.mm.free_high_bytes()
    }

    #[inline]
    pub fn checkpoint_vram(&self) -> VramCheckpoint {
        self.mm.checkpoint()
    }

    #[inline]
    pub fn restore_vram_checkpoint(&mut self, cp: VramCheckpoint) {
        self.mm.restore_checkpoint(cp);
    }

    pub fn poll_gsp_faults(&mut self) -> Result<(), String> {
        self.gsp.poll_faults()
    }

    /// Launch a single QMDv3 compute kernel and wait synchronously for timeline completion.
    pub fn launch_kernel_sync(
        &mut self,
        kern: &GpuKernel,
        grid: (u32, u32, u32),
        block: (u16, u16, u16),
        arg_bytes: &[u8],
        invalidate_caches: bool,
        timeout: Duration,
    ) -> Result<(), String> {
        self.timeline_value += 1;
        let sig_val = self.timeline_value;
        let sig_va = self.timeline_buf.va_addr;
        let (qmd_va, _) = self.prepare_qmd(kern, grid, block, arg_bytes, Some((sig_va, sig_val)));
        self.launch_qmd_chain(qmd_va, invalidate_caches)?;
        self.wait_timeline(sig_val, timeout)
    }

    /// Launch a barrier-guarded compute kernel, polling `err_ptr` for hardware watchdog trips
    /// and setting `cancel_ptr = 1` on timeout. Returns `Ok(0)` on success or `Ok(err_code)` if tripped.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kernel_watched(
        &mut self,
        kern: &GpuKernel,
        grid: (u32, u32, u32),
        block: (u16, u16, u16),
        arg_bytes: &[u8],
        invalidate_caches: bool,
        err_ptr: *const i32,
        cancel_ptr: *mut i32,
        timeout: Duration,
        label: &str,
    ) -> Result<i32, String> {
        self.timeline_value += 1;
        let sig_val = self.timeline_value;
        let sig_va = self.timeline_buf.va_addr;
        let (qmd_va, _) = self.prepare_qmd(kern, grid, block, arg_bytes, Some((sig_va, sig_val)));
        self.launch_qmd_chain(qmd_va, invalidate_caches)?;

        let sig_ptr = self.timeline_buf.as_ptr() as *const u64;
        let t_wait = Instant::now();
        let mut iter = 0u32;
        loop {
            let cur_sig = unsafe { std::ptr::read_volatile(sig_ptr) };
            if cur_sig >= sig_val || (sig_val < (1u64 << 31) && (cur_sig & 0xffff_ffff) >= sig_val) {
                return Ok(0);
            }
            let err = unsafe { std::ptr::read_volatile(err_ptr) };
            if err != 0 {
                return Ok(err);
            }
            iter = iter.wrapping_add(1);
            if (iter & 0x3fff) == 0 {
                self.poll_gsp_faults()?;
                if t_wait.elapsed() > Duration::from_millis(2) {
                    let _ = self.pci.ring_doorbell(self.compute_gpfifo.token);
                }
                if t_wait.elapsed() > timeout {
                    if !cancel_ptr.is_null() {
                        unsafe { std::ptr::write_volatile(cancel_ptr, 1) };
                    }
                    return Err(format!("{} timed out (>{}s)", label, timeout.as_secs()));
                }
            }
            std::hint::spin_loop();
        }
    }

    /// Attach timeline `release0` to the tail QMD of `chain`, hardware-link dependent QMDs,
    /// launch `chain[0]` with a single doorbell, and wait synchronously.
    pub fn launch_qmd_chain_sync(
        &mut self,
        chain: &[(u64, usize)],
        invalidate_caches: bool,
        timeout: Duration,
    ) -> Result<(), String> {
        if chain.is_empty() {
            return Ok(());
        }
        self.timeline_value += 1;
        let target_sig = self.timeline_value;
        let sig_va = self.timeline_buf.va_addr;
        {
            let (_, last_off) = *chain.last().unwrap();
            let buf = self.kernargs_buf_mut_slice_at(last_off);
            let mut tail_qmd = QmdV3 {
                bytes: buf[..256].try_into().unwrap(),
            };
            tail_qmd.set_release0(sig_va, target_sig);
            buf[..256].copy_from_slice(&tail_qmd.bytes);
        }
        for i in 0..(chain.len() - 1) {
            self.link_dependent_qmd(chain[i].1, chain[i + 1].0);
        }
        self.launch_qmd_chain(chain[0].0, invalidate_caches)?;
        self.wait_timeline(target_sig, timeout)
    }

    /// Launch a persistent multi-step megakernel asynchronously and return its opaque timeline ticket.
    pub fn launch_persistent_async(
        &mut self,
        kern: &GpuKernel,
        grid: (u32, u32, u32),
        block: (u16, u16, u16),
        arg_bytes: &[u8],
        invalidate_caches: bool,
    ) -> Result<u64, String> {
        self.timeline_value += 1;
        let sig_val = self.timeline_value;
        let sig_va = self.timeline_buf.va_addr;
        let (qmd_va, _) = self.prepare_qmd(kern, grid, block, arg_bytes, Some((sig_va, sig_val)));
        self.launch_qmd_chain(qmd_va, invalidate_caches)?;
        Ok(sig_val)
    }

    #[inline]
    pub fn is_ticket_done(&self, ticket: u64) -> bool {
        let sig_ptr = self.timeline_buf.as_ptr() as *const u64;
        let cur = unsafe { std::ptr::read_volatile(sig_ptr) };
        cur >= ticket || (ticket < (1u64 << 31) && (cur & 0xffff_ffff) >= ticket)
    }

    #[inline]
    pub fn poll_persistent_maintenance(&mut self, iter: &mut u32, nudge_doorbell: bool) -> Result<(), String> {
        *iter = iter.wrapping_add(1);
        if (*iter & 0x3fff) == 0 {
            self.poll_gsp_faults()?;
            if nudge_doorbell {
                let _ = self.pci.ring_doorbell(self.compute_gpfifo.token);
            }
        }
        Ok(())
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
        self.gsp.unload_guest_driver(&mut self.pci, Duration::from_secs(3));
        thread::sleep(Duration::from_millis(15));
    }
}

impl Drop for NvGpu {
    fn drop(&mut self) {
        self.shutdown_clean();
    }
}
