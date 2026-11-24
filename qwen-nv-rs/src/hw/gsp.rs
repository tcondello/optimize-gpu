//! GA102 GSP-RM RISC-V / Falcon Firmware Boot, RM RPCs, and Channel GPFIFO Setup.
#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{fence, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::defs::*;
use super::mmu::{align_down, align_up, VramBuffer, VramMemoryManager, MMU_CNT, MMU_COVERS};
use super::tinygpu::{SysmemMapping, TinyGpuClient};

#[inline]
pub(crate) unsafe fn ptr_read_u32_volatile(p: *const u8) -> u32 {
    unsafe { std::ptr::read_volatile(p as *const u32) }
}

#[inline]
pub(crate) unsafe fn ptr_write_u32_volatile(p: *mut u8, val: u32) {
    unsafe { std::ptr::write_volatile(p as *mut u32, val) };
}

// -----------------------------------------------------------------------------
// GSP Message Queue Ring & Firmware Boot State
// -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct FwsecDesc {
    pub(crate) stored_size: u32,
    pub(crate) pkc_data_off: u32,
    pub(crate) interface_off: u32,
    pub(crate) imem_pa: u32,
    pub(crate) imem_sz: u32,
    pub(crate) imem_va: u32,
    pub(crate) dmem_pa: u32,
    pub(crate) dmem_sz: u32,
    pub(crate) eng_id_mask: u16,
    pub(crate) ucode_id: u8,
}

pub(crate) struct GspTopologyInit {
    pub sm_version: u32,
    pub sass_version: u32,
    pub active_sms: u32,
    pub num_gpcs: u32,
    pub num_tpc_per_gpc: u32,
    pub num_sm_per_tpc: u32,
    pub max_warps_per_sm: u32,
    pub gpfifo_area: VramBuffer,
    pub compute_gpfifo: GpFifoRing,
    pub dma_gpfifo: GpFifoRing,
}

pub(crate) struct GspState {
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

pub(crate) struct GpFifoRing {
    pub bar1_ring_off: u64,
    pub bar1_gpput_off: u64,
    pub entries_count: u32,
    pub put_value: u32,
    pub token: u32,
}


impl GspState {
    pub(crate) fn wait_for_gfw_boot(pci: &mut TinyGpuClient) -> Result<(), String> {
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

    pub(crate) fn load_firmware_blobs() -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, [u32; 7]), String> {
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

    pub(crate) fn parse_vbios_fwsec(pci: &mut TinyGpuClient, vram_size: u64) -> Result<(SysmemMapping, FwsecDesc, u64), String> {
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

    pub(crate) fn prepare_gsp_sw(pci: &mut TinyGpuClient, vram_size: u64) -> Result<GspState, String> {
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

    pub(crate) fn flcn_reset(pci: &mut TinyGpuClient, base: u32, riscv: bool) -> Result<(), String> {
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

    pub(crate) fn flcn_disable_ctx_req(pci: &mut TinyGpuClient, base: u32) -> Result<(), String> {
        let ctl = pci.rreg(base + FLCN_FBIF_CTL)?;
        pci.wreg(base + FLCN_FBIF_CTL, ctl | (1 << 7))?; // allow_phys_no_ctx = 1
        pci.wreg(base + FLCN_DMACTL, 0)
    }

    pub(crate) fn flcn_start_cpu(pci: &mut TinyGpuClient, base: u32) -> Result<(), String> {
        let cpuctl = pci.rreg(base + FLCN_CPUCTL)?;
        if ((cpuctl >> 6) & 1) == 1 {
            pci.wreg(base + FLCN_CPUCTL_ALIAS, 0x2)
        } else {
            pci.wreg(base + FLCN_CPUCTL, 0x2) // startcpu = bit 1
        }
    }

    pub(crate) fn flcn_wait_halted(pci: &mut TinyGpuClient, base: u32) -> Result<(), String> {
        let t0 = Instant::now();
        while ((pci.rreg(base + FLCN_CPUCTL)? >> 4) & 1) == 0 {
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(format!("Falcon {:#x} did not halt within 10s", base));
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    pub(crate) fn flcn_execute_dma(
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
    pub(crate) fn flcn_execute_hs(
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

    pub(crate) fn boot_falcons_and_gsp(pci: &mut TinyGpuClient, gsp: &mut GspState) -> Result<(), String> {
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

    pub(crate) fn gsp_checksum(data: &[u8]) -> u32 {
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

    pub(crate) fn gsp_send_record(pci: &mut TinyGpuClient, gsp: &mut GspState, func: u32, payload: &[u8]) -> Result<(), String> {
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

    pub(crate) fn gsp_send_rpc(pci: &mut TinyGpuClient, gsp: &mut GspState, func: u32, payload: &[u8]) -> Result<(), String> {
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

    pub(crate) fn gsp_run_cpu_seq(pci: &mut TinyGpuClient, gsp: &GspState, seq_buf: &[u8]) -> Result<(), String> {
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

    pub(crate) fn gsp_wait_resp(
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

    pub(crate) fn alloc_handle(gsp: &mut GspState) -> u32 {
        let h = gsp.next_handle;
        gsp.next_handle += 1;
        h
    }

    pub(crate) fn rpc_rm_control(
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

    pub(crate) fn rpc_set_page_directory(
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

    pub(crate) fn promote_ctx(
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

    pub(crate) fn rpc_rm_alloc(
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

    pub(crate) fn init_golden_image(
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
    pub(crate) fn create_gpfifo(
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


    pub(crate) fn init_user_channels_and_topology(
        &mut self,
        pci: &mut TinyGpuClient,
        mm: &mut VramMemoryManager,
    ) -> Result<GspTopologyInit, String> {
        let user_root = self.user_root;
        let root_params = [0u8; 120];
        Self::rpc_rm_alloc(pci, mm, self, 0, NV01_ROOT, &root_params, user_root)?;

        let mut dev_params = [0u8; 56];
        put_u32(&mut dev_params, 0, 0);
        put_u32(&mut dev_params, 4, user_root);
        put_u32(&mut dev_params, 48, 0);
        let user_device = Self::rpc_rm_alloc(pci, mm, self, user_root, NV01_DEVICE_0, &dev_params, user_root)?;
        self.user_device = user_device;

        let subdev_params = [0u8; 4];
        let user_subdevice = Self::rpc_rm_alloc(pci, mm, self, user_device, NV20_SUBDEVICE_0, &subdev_params, user_root)?;
        self.user_subdevice = user_subdevice;

        let mut virt_params = [0u8; 24];
        put_u64(&mut virt_params, 8, 0x0001_ffff_ffff_ffff);
        let virtmem = Self::rpc_rm_alloc(pci, mm, self, user_device, NV01_MEMORY_VIRTUAL, &virt_params, user_root)?;
        self.user_virtmem = virtmem;

        if std::env::var("QWEN_NV_MAX_BOOST").ok().as_deref() == Some("1") {
            let mut boost_params = [0u8; 8];
            put_u32(&mut boost_params, 0, NV2080_PERF_BOOST_FLAGS_MAX);
            put_u32(&mut boost_params, 4, 0xffff_ffff);
            let _ = Self::rpc_rm_control(pci, self, user_subdevice, NV2080_CTRL_CMD_PERF_BOOST, &boost_params, user_root)?;
        }

        let mut va_params = [0u8; 48];
        put_u32(&mut va_params, 4, NV_VASPACE_FLAGS_USER);
        put_u64(&mut va_params, 8, 0x0001_ffff_fb00_0000);
        put_u64(&mut va_params, 40, 0x1000);
        let user_vaspace = Self::rpc_rm_alloc(pci, mm, self, user_device, FERMI_VASPACE_A, &va_params, user_root)?;
        Self::rpc_set_page_directory(pci, self, user_device, user_vaspace, mm.root_pt_paddr, user_root)?;

        let mut chgrp_params = [0u8; 20];
        put_u32(&mut chgrp_params, 12, NV2080_ENGINE_TYPE_GRAPHICS);
        let channel_group = Self::rpc_rm_alloc(pci, mm, self, user_device, KEPLER_CHANNEL_GROUP_A, &chgrp_params, user_root)?;

        let gpfifo_pa = mm.palloc_bar1(pci, 0x30_0000, 0x10000, true)?;
        let gpfifo_va = mm.alloc_vaddr(0x30_0000, 0x10000);
        mm.map_range(pci, gpfifo_va, &[(gpfifo_pa, 0x30_0000)], false, false)?;
        let gpfifo_area = VramBuffer {
            va_addr: gpfifo_va,
            size: 0x30_0000,
            paddrs: vec![(gpfifo_pa, 0x30_0000)],
        };

        let mut ctxshare_params = [0u8; 12];
        put_u32(&mut ctxshare_params, 0, user_vaspace);
        put_u32(&mut ctxshare_params, 4, NV_CTXSHARE_FLAGS_ASYNC);
        let ctxshare = Self::rpc_rm_alloc(pci, mm, self, channel_group, FERMI_CONTEXT_SHARE_A, &ctxshare_params, user_root)?;
        self.user_ctxshare = ctxshare;

        let compute_gpfifo = Self::create_gpfifo(pci, mm, self, &gpfifo_area, ctxshare, channel_group, 0, 0x10000, true)?;
        let dma_gpfifo = Self::create_gpfifo(pci, mm, self, &gpfifo_area, ctxshare, channel_group, 0x10_0000, 0x10000, false)?;

        let mut sched_params = [0u8; 2];
        sched_params[0] = 1;
        let _ = Self::rpc_rm_control(pci, self, channel_group, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, &sched_params, user_root)?;

        let gr_info_req = vec![0u8; 3712];
        let gr_info_resp = Self::rpc_rm_control(pci, self, user_subdevice, NV2080_CTRL_CMD_KGR_GET_INFO, &gr_info_req, user_root)?;
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
            if let Ok(tpc_resp) = Self::rpc_rm_control(pci, self, user_subdevice, NV2080_CTRL_CMD_GR_GET_TPC_MASK, &tpc_req, user_root) {
                let mask = get_u32(&tpc_resp, 20);
                active_tpcs += mask.count_ones();
            }
        }
        let active_sms = if active_tpcs > 0 { active_tpcs * num_sm_per_tpc } else { 68 };

        Ok(GspTopologyInit {
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
        })
    }

    pub(crate) fn unload_guest_driver(&mut self, pci: &mut TinyGpuClient, timeout: Duration) {
        let mut unload_req = [0u8; 8];
        put_u32(&mut unload_req, 4, 1 << 6); // __GPU_STATE_FLAGS_FAST_UNLOAD
        let ok = Self::gsp_send_rpc(pci, self, NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, &unload_req)
            .and_then(|_| Self::gsp_wait_resp(pci, self, NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, timeout))
            .is_ok();
        if !ok {
            Self::emergency_halt_falcons(pci);
        }
    }

    pub(crate) fn emergency_halt_falcons(pci: &mut TinyGpuClient) {
        let _ = pci.wreg(NV_PGSP_FALCON_ENGINE, 1);
        let _ = pci.wreg(NV_PSEC_FALCON_ENGINE, 1);
        thread::sleep(Duration::from_millis(25));
    }

    pub(crate) fn poll_faults(&mut self) -> Result<(), String> {
        let rx_ptr_addr = unsafe { self.queues_mem.ptr.add(self.cmd_q_off + 32) };
        let stat_wp_addr = unsafe { self.queues_mem.ptr.add(self.stat_q_off + 16) };
        let stat_ring_base = self.stat_q_off + 0x1000;
        let ring_cap = (self.msg_count as usize) * self.msg_size;

        let mut rp = unsafe { ptr_read_u32_volatile(rx_ptr_addr) };
        let wp = unsafe { ptr_read_u32_volatile(stat_wp_addr) };
        if rp == wp {
            return Ok(());
        }

        let mut fault_err: Option<String> = None;
        while rp != wp {
            let off = (rp as usize) * self.msg_size;
            let hdr_off = stat_ring_base + off + 0x30;
            let func = get_u32(self.queues_mem.as_slice(), hdr_off + 12);
            let rpc_len = get_u32(self.queues_mem.as_slice(), hdr_off + 8) as usize;
            let payload_len = rpc_len.saturating_sub(32);
            let mut msg = vec![0u8; payload_len];
            let msg_ring_off = off + 0x50;
            if msg_ring_off + payload_len <= ring_cap {
                msg.copy_from_slice(&self.queues_mem.as_slice()[stat_ring_base + msg_ring_off .. stat_ring_base + msg_ring_off + payload_len]);
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
            let elems = ((rpc_len + self.msg_size - 1) / self.msg_size).max(1) as u32;
            rp = (rp + elems) % self.msg_count;
            unsafe { ptr_write_u32_volatile(rx_ptr_addr, rp) };
        }
        fence(Ordering::SeqCst);
        if let Some(e) = fault_err {
            Err(e)
        } else {
            Ok(())
        }
    }
}
