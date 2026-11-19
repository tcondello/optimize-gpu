//! Exact GA102 (Ampere sm_86) Hardware Register, MMU, GSP-RM, and QMDv3 Definitions.
//! Zero runtime parsing, zero Python/C header dependencies.

#![allow(dead_code)]

// -----------------------------------------------------------------------------
// TinyGPU Unix Socket Protocol Opcodes (`RemoteCmd`)
// -----------------------------------------------------------------------------
pub const REMOTE_CMD_PROBE: u8        = 0;
pub const REMOTE_CMD_MAP_BAR: u8      = 1;
pub const REMOTE_CMD_MAP_SYSMEM_FD: u8= 2;
pub const REMOTE_CMD_CFG_READ: u8     = 3;
pub const REMOTE_CMD_CFG_WRITE: u8    = 4;
pub const REMOTE_CMD_RESET: u8        = 5;
pub const REMOTE_CMD_MMIO_READ: u8    = 6;
pub const REMOTE_CMD_MMIO_WRITE: u8   = 7;
pub const REMOTE_CMD_MAP_SYSMEM: u8   = 8;
pub const REMOTE_CMD_SYSMEM_READ: u8  = 9;
pub const REMOTE_CMD_SYSMEM_WRITE: u8 = 10;
pub const REMOTE_CMD_RESIZE_BAR: u8   = 11;
pub const REMOTE_CMD_PING: u8         = 12;

// -----------------------------------------------------------------------------
// PCI Configuration Space Offsets
// -----------------------------------------------------------------------------
pub const PCI_VENDOR_ID: u64            = 0x00;
pub const PCI_COMMAND: u64              = 0x04;
pub const PCI_COMMAND_MASTER: u64       = 0x04;
pub const PCI_REVISION_ID: u64          = 0x08;
pub const PCI_SUBSYSTEM_VENDOR_ID: u64  = 0x2c;

// -----------------------------------------------------------------------------
// BAR0 MMIO Register Offsets (GA102)
// -----------------------------------------------------------------------------
pub const NV_PMC_BOOT_0: u32                                        = 0x000000;
pub const NV_PMC_ENABLE: u32                                        = 0x000200;
pub const NV_PMC_DEVICE_ENABLE: u32                                 = 0x000600;
pub const NV_PMC_BOOT_42: u32                                       = 0x000a00;
pub const NV_PBUS_BAR1_BLOCK: u32                                   = 0x001704;
pub const NV_PGSP_FALCON_BASE: u32                                  = 0x110000;
pub const NV_PSEC_FALCON_BASE: u32                                  = 0x840000;
pub const NV_PGSP_FALCON_MAILBOX0: u32                              = 0x110040;
pub const NV_PGSP_FALCON_MAILBOX1: u32                              = 0x110044;
pub const NV_PGSP_FALCON_ENGINE: u32                                = 0x1103c0;
pub const NV_PGSP_QUEUE_HEAD_0: u32                                 = 0x110c00;
pub const NV_PSEC_FALCON_ENGINE: u32                                = 0x8403c0;
pub const NV_PGC6_BSI_SECURE_SCRATCH_14: u32                        = 0x1180f8;
pub const NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK: u32  = 0x118128;
pub const NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0: u32                = 0x118234;
pub const NV_PGC6_AON_SECURE_SCRATCH_GROUP_42: u32                  = 0x1183a4;
pub const NV_PFB_PRI_MMU_WPR2_ADDR_HI: u32                          = 0x1fa828;
pub const NV_PROM_BASE: u32                                         = 0x300000;
pub const NV_PROM_SIZE: usize                                       = 0x100000;
pub const NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE: u32              = 0xb830b0;
pub const NV_USERMODE_DOORBELL_BAR0_OFF: u64                        = 0xbb0090;

// Falcon Relative Offsets
pub const FLCN_MAILBOX0: u32            = 0x0040;
pub const FLCN_MAILBOX1: u32            = 0x0044;
pub const FLCN_OS: u32                  = 0x0080;
pub const FLCN_RM: u32                  = 0x0084;
pub const FLCN_HWCFG2: u32              = 0x00f4;
pub const FLCN_CPUCTL: u32              = 0x0100;
pub const FLCN_BOOTVEC: u32             = 0x0104;
pub const FLCN_DMACTL: u32              = 0x010c;
pub const FLCN_DMATRFBASE: u32          = 0x0110;
pub const FLCN_DMATRFMOFFS: u32         = 0x0114;
pub const FLCN_DMATRFCMD: u32           = 0x0118;
pub const FLCN_DMATRFFBOFFS: u32        = 0x011c;
pub const FLCN_DMATRFBASE1: u32         = 0x0128;
pub const FLCN_CPUCTL_ALIAS: u32        = 0x0130;
pub const FLCN_FBIF_TRANSCFG0: u32      = 0x0600;
pub const FLCN_FBIF_CTL: u32            = 0x0624;
pub const FLCN2_MOD_SEL: u32            = 0x1180;
pub const FLCN2_CURR_UCODE_ID: u32      = 0x1198;
pub const FLCN2_ENGIDMASK: u32          = 0x119c;
pub const FLCN2_BROM_PARAADDR0: u32     = 0x1210;
pub const RISCV_CPUCTL: u32             = 0x1388;
pub const RISCV_BCR_CTRL: u32           = 0x1668;

// -----------------------------------------------------------------------------
// VBIOS / FWSEC Constants
// -----------------------------------------------------------------------------
pub const OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR: usize = 0x18;
pub const OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN: usize       = 0x10;
pub const PCI_ROM_IMAGE_BLOCK_SIZE: usize                 = 0x200;
pub const OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE: usize       = 0x14;
pub const NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE: u8 = 0x00;
pub const NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT: u8  = 0xe0;
pub const BIT_TOKEN_FALCON_DATA: u8                       = 0x70;
pub const BIT_DATA_FALCON_DATA_V2_SIZE_4: u16             = 4;
pub const FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: u8         = 0x85;
pub const FALCON_UCODE_DESC_V3_SIZE_44: usize             = 44;
pub const FALCON_APPIF_ENTRY_ID_DMEMMAPPER: u32           = 0x4;

// -----------------------------------------------------------------------------
// GSP-RM & LIBOS Constants
// -----------------------------------------------------------------------------
pub const LIBOS_MEMORY_REGION_CONTIGUOUS: u8              = 1;
pub const LIBOS_MEMORY_REGION_LOC_SYSMEM: u8              = 1;
pub const LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2: u32        = 12;
pub const GSP_FW_WPR_META_REVISION: u64                   = 1;
pub const GSP_FW_WPR_META_MAGIC: u64                      = 0xdc3aae21371a60b3;

pub const NV_VGPU_MSG_SIGNATURE_VALID: u32                = 0x43505256;
pub const NV_VGPU_MSG_RESULT_RPC_PENDING: u32             = 0xffffffff;
pub const NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY: u32    = 0x36;
pub const NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER: u32= 0x2f;
pub const NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD: u32   = 0x47;
pub const NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO: u32   = 0x48;
pub const NV_VGPU_MSG_FUNCTION_SET_REGISTRY: u32          = 0x49;
pub const NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL: u32        = 0x4c;
pub const NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC: u32          = 0x67;

pub const NV_VGPU_MSG_EVENT_GSP_INIT_DONE: u32            = 0x1001;
pub const NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: u32    = 0x1002;
pub const NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED: u32         = 0x1005;
pub const NV_VGPU_MSG_EVENT_OS_ERROR_LOG: u32             = 0x1006;

pub const REGISTRY_TABLE_ENTRY_TYPE_DWORD: u8             = 1;

// RM Object Classes & Control IDs
pub const NV01_ROOT: u32                                  = 0x0;
pub const NV01_DEVICE_0: u32                              = 0x80;
pub const NV01_MEMORY_VIRTUAL: u32                        = 0x70;
pub const NV20_SUBDEVICE_0: u32                           = 0x2080;
pub const GT200_DEBUGGER: u32                             = 0x83de;
pub const FERMI_CONTEXT_SHARE_A: u32                      = 0x9067;
pub const FERMI_VASPACE_A: u32                            = 0x90f1;
pub const KEPLER_CHANNEL_GROUP_A: u32                     = 0xa06c;
pub const AMPERE_CHANNEL_GPFIFO_A: u32                    = 0xc56f;
pub const AMPERE_DMA_COPY_B: u32                          = 0xc7b5;
pub const AMPERE_COMPUTE_B: u32                           = 0xc7c0;

pub const NV_VASPACE_FLAGS_USER: u32                      = 0x48; // ENABLE_PAGE_FAULTING(0x40) | EXTERNALLY_OWNED(0x8)
pub const NV2080_ENGINE_TYPE_GRAPHICS: u32                = 0x1;
pub const NV_CTXSHARE_FLAGS_ASYNC: u32                    = 0x1;
pub const NV2080_CTRL_CMD_GPU_PROMOTE_CTX: u32            = 0x2080012b;
pub const NV2080_CTRL_CMD_KGR_GET_INFO: u32               = 0x20800a2a;
pub const NV2080_CTRL_CMD_KGR_GET_CTX_BUFS_INFO: u32      = 0x20800a32;
pub const NV2080_CTRL_CMD_GR_GET_TPC_MASK: u32            = 0x2080122b;
pub const NV2080_CTRL_CMD_PERF_BOOST: u32                 = 0x2080200a;
pub const NV2080_PERF_BOOST_FLAGS_MAX: u32                = 0x52;
pub const NV90F1_CTRL_CMD_COPY_SERVER_RESERVED_PDES: u32  = 0x90f10106;
pub const NVA06C_CTRL_CMD_GPFIFO_SCHEDULE: u32            = 0xa06c0101;
pub const NVC36F_CTRL_CMD_GET_WORK_SUBMIT_TOKEN: u32      = 0xc36f0108;

pub const GR_INFO_SM_VERSION: usize                       = 0x0c;
pub const GR_INFO_MAX_WARPS_PER_SM: usize                 = 0x0d;
pub const GR_INFO_NUM_GPCS: usize                         = 0x14;
pub const GR_INFO_NUM_TPC_PER_GPC: usize                  = 0x17;
pub const GR_INFO_NUM_SM_PER_TPC: usize                   = 0x20;

// GPFIFO Pushbuffer Methods
pub const NVC56F_NON_STALL_INTERRUPT: u32                 = 0x0020;
pub const NVC56F_SEM_ADDR_LO: u32                         = 0x005c;
pub const NVC56F_SEM_EXEC_ACQ_GEQ_64: u32                 = 0x0100_0003;
pub const NVC56F_SEM_EXEC_REL_WFI_64_TS: u32              = 0x0310_0001;
pub const GPPUT_CTRL_OFFSET: usize                        = 140; // AmpereAControlGPFifo.GPPut

pub const NVC6C0_SET_OBJECT: u32                          = 0x0000;
pub const NVC6C0_SET_SHARED_MEM_WINDOW_A: u32             = 0x02a0;
pub const NVC6C0_SEND_PCAS_A: u32                         = 0x02b4;
pub const NVC6C0_SEND_SIGNALING_PCAS2_B: u32              = 0x02c0;
pub const NVC6C0_SET_LOCAL_MEM_NON_THROTTLED_A: u32       = 0x02e4;
pub const NVC6C0_SET_LOCAL_MEM_A: u32                     = 0x0790;
pub const NVC6C0_SET_LOCAL_MEM_WINDOW_A: u32              = 0x07b0;
pub const NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI: u32     = 0x1698;
pub const NVC6C0_INVALIDATE_FLAGS: u32                    = 0x1011;

pub const NVC6B5_SET_SEMAPHORE_A: u32                     = 0x0240;
pub const NVC6B5_LAUNCH_DMA: u32                          = 0x0300;
pub const NVC6B5_LAUNCH_DMA_COPY_FLAGS: u32               = 0x0182;
pub const NVC6B5_LAUNCH_DMA_SIG_FLAGS: u32                = 0x0014;
pub const NVC6B5_OFFSET_IN_UPPER: u32                     = 0x0400;
pub const NVC6B5_LINE_LENGTH_IN: u32                      = 0x0418;

// -----------------------------------------------------------------------------
// Turing/Ampere MMU v2 PTE / PDE / Dual-PDE Bit Encoders
// -----------------------------------------------------------------------------
#[inline]
pub fn encode_mmu_v2_pte(paddr: u64, is_sys: bool, uncached: bool, valid: bool) -> u64 {
    if !valid { return 0; }
    let valid_b = 1u64;
    let aperture = if is_sys { 2u64 } else { 0u64 }; // bits 1..=2
    let vol = if uncached { 1u64 } else { 0u64 };    // bit 3
    // Note: in NVDev.set_entry: `address_sys = paddr >> 12`, field `address_sys` starts at bit 8!
    let addr = (paddr >> 12) << 8;
    let kind = 6u64 << 56;
    valid_b | (aperture << 1) | (vol << 3) | addr | kind
}

#[inline]
pub fn encode_mmu_v2_pde(paddr: u64, valid: bool) -> u64 {
    if !valid { return 0; }
    let aperture = 1u64 << 1; // video mem page table
    let no_ats = 1u64 << 5;
    let addr = (paddr >> 12) << 8;
    aperture | no_ats | addr
}

#[inline]
pub fn encode_mmu_v2_dual_pde_small(paddr: u64, valid: bool) -> (u64, u64) {
    if !valid { return (0, 0); }
    // `no_ats` bit 5 (lo word), `aperture_small` bits 65..=66 -> bit 1 of hi word,
    // `address_small_sys` bits 72..=117 -> bit 8 of hi word!
    let lo = 1u64 << 5;
    let hi = (1u64 << 1) | ((paddr >> 12) << 8);
    (lo, hi)
}

// -----------------------------------------------------------------------------
// QMDv03_00 (256-byte Ampere Compute Queue Meta Data) Bit Packer
// -----------------------------------------------------------------------------
#[derive(Clone)]
pub struct QmdV3 {
    pub bytes: [u8; 256],
}

impl QmdV3 {
    pub fn new_zeroed() -> Self {
        Self { bytes: [0u8; 256] }
    }

    #[inline]
    pub fn write_bits(&mut self, hi: usize, lo: usize, val: u64) {
        let width = hi - lo + 1;
        let mut cur: u128 = 0;
        let start_b = lo / 8;
        let end_b = hi / 8;
        for (i, b_idx) in (start_b..=end_b).enumerate() {
            cur |= (self.bytes[b_idx] as u128) << (i * 8);
        }
        let shift = lo % 8;
        let mask: u128 = if width == 128 { u128::MAX } else { ((1u128 << width) - 1) << shift };
        cur = (cur & !mask) | (((val as u128) << shift) & mask);
        for (i, b_idx) in (start_b..=end_b).enumerate() {
            self.bytes[b_idx] = ((cur >> (i * 8)) & 0xff) as u8;
        }
    }

    pub fn init_ampere_template(
        &mut self,
        prog_addr: u64,
        prog_sz: usize,
        shmem_bytes: u32,
        regs_usage: u32,
        slm_per_thread: u32,
        sass_version: u32,
        cbuf0_addr: u64,
        cbuf0_size: u32,
    ) {
        let smem_kb = if shmem_bytes <= 32 * 1024 {
            32 * 1024
        } else if shmem_bytes <= 64 * 1024 {
            64 * 1024
        } else {
            100 * 1024
        };
        let smem_cfg = (smem_kb / 4096 + 1) as u64;

        self.write_bits(583, 580, 3); // QMD_MAJOR_VERSION = 3
        self.write_bits(134, 134, 1); // SM_GLOBAL_CACHING_ENABLE = 1
        self.write_bits(1584, 1568, prog_addr >> 32);
        self.write_bits(1567, 1536, prog_addr & 0xffff_ffff);
        self.write_bits(561, 544, shmem_bytes as u64);
        self.write_bits(656, 648, regs_usage as u64);
        self.write_bits(1623, 1600, slm_per_thread as u64);
        self.write_bits(133, 128, 0x3f); // QMD_GROUP_ID
        self.write_bits(186, 186, 1);    // INVALIDATE_TEXTURE_HEADER_CACHE
        self.write_bits(187, 187, 1);    // INVALIDATE_TEXTURE_SAMPLER_CACHE
        self.write_bits(188, 188, 1);    // INVALIDATE_TEXTURE_DATA_CACHE
        self.write_bits(189, 189, 1);    // INVALIDATE_SHADER_DATA_CACHE
        self.write_bits(378, 378, 1);    // API_VISIBLE_CALL_LIMIT
        self.write_bits(382, 382, 1);    // SAMPLER_INDEX
        self.write_bits(767, 763, 1);    // BARRIER_COUNT
        // Thunderbolt PCIe optimization: L1_MEMBAR (3) instead of L1_SYSMEMBAR (1)!
        self.write_bits(369, 368, 3);    // CWD_MEMBAR_TYPE = L1_MEMBAR
        self.write_bits(1074, 1074, 1);  // CONSTANT_BUFFER_INVALIDATE_0
        self.write_bits(567, 562, smem_cfg);
        self.write_bits(662, 657, smem_cfg);
        self.write_bits(574, 569, 0x1a); // MAX_SM_CONFIG_SHARED_MEM_SIZE
        self.write_bits(1649, 1641, ((prog_sz >> 8).min(0x1ff)) as u64);
        self.write_bits(1663, 1656, sass_version as u64);
        self.write_bits(1640, 1632, prog_addr >> 40);
        self.write_bits(287, 256, (prog_addr >> 8) & 0xffff_ffff);

        self.set_cbuf(0, cbuf0_addr, cbuf0_size);
    }

    pub fn set_cbuf(&mut self, idx: usize, addr: u64, size: u32) {
        // Each cbuf entry in QMDv3 is spaced by 64 bits from CONSTANT_BUFFER_ADDR_LOWER_0 (1024)
        let base_lo = 1024 + idx * 64;
        self.write_bits(base_lo + 31, base_lo, addr & 0xffff_ffff);
        self.write_bits(base_lo + 48, base_lo + 32, (addr >> 32) & 0x1ffff);
        self.write_bits(base_lo + 63, base_lo + 51, size as u64); // note: tinygrad passes raw sz to shifted4 field
        self.write_bits(640 + idx, 640 + idx, 1);                 // CONSTANT_BUFFER_VALID_i
    }

    pub fn set_launch_dims(&mut self, grid: (u32, u32, u32), block: (u16, u16, u16)) {
        self.write_bits(415, 384, grid.0 as u64);
        self.write_bits(431, 416, grid.1 as u64);
        self.write_bits(463, 448, grid.2 as u64);
        self.write_bits(607, 592, block.0 as u64);
        self.write_bits(623, 608, block.1 as u64);
        self.write_bits(639, 624, block.2 as u64);
    }

    pub fn set_dependent_qmd0(&mut self, next_qmd_addr: u64) {
        self.write_bits(511, 480, next_qmd_addr >> 8);
        self.write_bits(515, 513, 1); // ACTION = 1
        self.write_bits(516, 516, 1); // PREFETCH = 1
        self.write_bits(512, 512, 1); // ENABLE = 1
    }

    pub fn set_release0(&mut self, sig_addr: u64, payload: u64) {
        self.write_bits(823, 823, 1);    // RELEASE0_ENABLE
        self.write_bits(829, 829, 1);    // RELEASE0_PAYLOAD64B = TRUE
        self.write_bits(819, 819, 1);    // RELEASE0_MEMBAR_TYPE = FE_SYSMEMBAR (ensures host coherence!)
        self.write_bits(799, 768, sig_addr & 0xffff_ffff);
        self.write_bits(807, 800, (sig_addr >> 32) & 0xff);
        self.write_bits(863, 832, payload & 0xffff_ffff);
        self.write_bits(895, 864, payload >> 32);
    }
}

// Little-endian byte slice helpers for exact-offset struct packing
#[inline] pub fn put_u8(buf: &mut [u8], off: usize, v: u8) { buf[off] = v; }
#[inline] pub fn put_u16(buf: &mut [u8], off: usize, v: u16) { buf[off..off+2].copy_from_slice(&v.to_le_bytes()); }
#[inline] pub fn put_u32(buf: &mut [u8], off: usize, v: u32) { buf[off..off+4].copy_from_slice(&v.to_le_bytes()); }
#[inline] pub fn put_u64(buf: &mut [u8], off: usize, v: u64) { buf[off..off+8].copy_from_slice(&v.to_le_bytes()); }
#[inline] pub fn get_u16(buf: &[u8], off: usize) -> u16 { u16::from_le_bytes(buf[off..off+2].try_into().unwrap()) }
#[inline] pub fn get_u32(buf: &[u8], off: usize) -> u32 { u32::from_le_bytes(buf[off..off+4].try_into().unwrap()) }
#[inline] pub fn get_u64(buf: &[u8], off: usize) -> u64 { u64::from_le_bytes(buf[off..off+8].try_into().unwrap()) }
