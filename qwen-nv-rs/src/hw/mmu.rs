//! 5-Level Turing/Ampere MMU v2 Page Table & Physical GDDR6X Manager.
#![allow(dead_code)]

use std::collections::HashMap;
use super::defs::*;
use super::tinygpu::{SysmemMapping, TinyGpuClient};

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
    pub(crate) paddrs: Vec<(u64, u64)>,
}

pub struct HostMappedBuffer {
    pub va_addr: u64,
    pub size: usize,
    mapping: SysmemMapping,
}

impl HostMappedBuffer {
    #[inline]
    pub(super) fn new(va_addr: u64, size: usize, mapping: SysmemMapping) -> Self {
        Self { va_addr, size, mapping }
    }

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

pub(super) const MMU_COVERS: [u64; 5] = [1 << 47, 1 << 38, 1 << 29, 1 << 21, 1 << 12];
pub(super) const MMU_CNT: [usize; 5]  = [4, 512, 512, 256, 512]; // lv=3 is Dual-PDE (256 * 16B = 4096B)

struct PtPage {
    words: [u64; 512],
    dirty: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct VramCheckpoint {
    high_pa_next: u64,
    bar1_pa_next: u64,
}

pub(crate) struct VramMemoryManager {
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

    pub fn free_high_bytes(&self) -> u64 {
        self.high_pa_end.saturating_sub(self.high_pa_next)
    }

    pub fn checkpoint(&self) -> VramCheckpoint {
        VramCheckpoint {
            high_pa_next: self.high_pa_next,
            bar1_pa_next: self.bar1_pa_next,
        }
    }

    pub fn restore_checkpoint(&mut self, cp: VramCheckpoint) {
        self.high_pa_next = cp.high_pa_next;
        self.bar1_pa_next = cp.bar1_pa_next;
        // Never rewind `va_next`: 48-bit VA space (`256 TiB`) gives every new allocation a fresh
        // 2 MiB-aligned virtual address, preventing Dual-PDE leaf/table collisions and stale TLBs!
        self.va_next = align_up(self.va_next, 2 << 20);
    }
}
