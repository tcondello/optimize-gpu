//! Native macOS ARM64 Client for `/Applications/TinyGPU.app` (`tinygpu.sock`).
//! Connects directly via Unix domain socket + `SCM_RIGHTS` shared-memory `mmap`
//! with zero Python/tinygrad middleware.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::ptr;
use std::thread;
use std::time::Duration;

use crate::nv_defs::*;

// macOS Darwin POSIX constants & FFI (`libSystem.B.dylib` is linked by `std`)
const SOL_SOCKET: i32 = 0xffff;
const SCM_RIGHTS: i32 = 0x01;
const PROT_READ: i32 = 0x01;
const PROT_WRITE: i32 = 0x02;
const MAP_SHARED: i32 = 0x0001;
const MAP_FAILED_ISIZE: isize = -1;
const LOCK_EX: i32 = 0x02;
const LOCK_NB: i32 = 0x04;
const LOCK_UN: i32 = 0x08;

#[repr(C)]
struct IoVec {
    iov_base: *mut u8,
    iov_len: usize,
}

#[repr(C)]
struct MsgHdr {
    msg_name: *mut u8,
    msg_namelen: u32,
    msg_iov: *mut IoVec,
    msg_iovlen: i32,
    msg_control: *mut u8,
    msg_controllen: u32,
    msg_flags: i32,
}

#[repr(C)]
struct CMsgHdr {
    cmsg_len: u32,
    cmsg_level: i32,
    cmsg_type: i32,
}

extern "C" {
    fn recvmsg(socket: RawFd, msg: *mut MsgHdr, flags: i32) -> isize;
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: RawFd, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn close(fd: RawFd) -> i32;
    fn flock(fd: RawFd, operation: i32) -> i32;
    fn setsockopt(socket: RawFd, level: i32, option_name: i32, option_value: *const u8, option_len: u32) -> i32;
}

const SO_SNDBUF: i32 = 0x1001;
const SO_RCVBUF: i32 = 0x1002;

pub struct SysmemMapping {
    pub ptr: *mut u8,
    pub mapped_size: usize,
    pub req_size: usize,
    pub paddrs_4k: Vec<u64>,
}

unsafe impl Send for SysmemMapping {}
unsafe impl Sync for SysmemMapping {}

impl SysmemMapping {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.req_size) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.req_size) }
    }
}

impl Drop for SysmemMapping {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.mapped_size > 0 {
            unsafe {
                munmap(self.ptr, self.mapped_size);
            }
            self.ptr = ptr::null_mut();
        }
    }
}

pub struct TinyGpuClient {
    sock: UnixStream,
    dev_id: u32,
    lock_file: File,
    bar_cache: [(u64, u64); 6],
}

impl TinyGpuClient {
    pub fn connect() -> Result<Self, String> {
        let tmp_dir = std::env::temp_dir();
        let sock_path = std::env::var("APL_REMOTE_SOCK")
            .map(PathBuf::from)
            .unwrap_or_else(|_| tmp_dir.join("tinygpu.sock"));

        let app_bin = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";
        let mut sock_opt = None;
        for attempt in 0..100 {
            if let Ok(s) = UnixStream::connect(&sock_path) {
                sock_opt = Some(s);
                break;
            }
            if attempt == 0 {
                let _ = Command::new(app_bin)
                    .arg("server")
                    .arg(&sock_path)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
            }
            thread::sleep(Duration::from_millis(50));
        }

        let sock = sock_opt.ok_or_else(|| {
            format!("Failed to connect to TinyGPU server at {}", sock_path.display())
        })?;
        let _ = sock.set_read_timeout(Some(Duration::from_secs(15)));
        let _ = sock.set_write_timeout(Some(Duration::from_secs(15)));

        // Enlarge socket buffers to 64 MB
        unsafe {
            let sz: i32 = 64 << 20;
            setsockopt(sock.as_raw_fd(), SOL_SOCKET, SO_SNDBUF, &sz as *const i32 as *const u8, 4);
            setsockopt(sock.as_raw_fd(), SOL_SOCKET, SO_RCVBUF, &sz as *const i32 as *const u8, 4);
        }

        // Acquire exclusive lock `nv_usb4.lock` matching tinygrad
        let lock_path = tmp_dir.join("nv_usb4.lock");
        let lock_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|e| format!("Failed to open {}: {}", lock_path.display(), e))?;

        let rc = unsafe { flock(lock_file.as_raw_fd(), LOCK_EX | LOCK_NB) };
        if rc != 0 {
            return Err(format!(
                "NVIDIA eGPU lock {} is currently held by another process (stop existing daemon/process first)",
                lock_path.display()
            ));
        }

        let mut client = Self {
            sock,
            dev_id: 0,
            lock_file,
            bar_cache: [(0, 0); 6],
        };

        // Verify connection & NVIDIA vendor ID via `CFG_READ(PCI_VENDOR_ID, 4)`
        let vid_did = client.read_config(PCI_VENDOR_ID, 4)?;
        if (vid_did & 0xffff) != 0x10de {
            return Err(format!("Unexpected PCI vendor/device ID {:#x} (expected 0x10de)", vid_did));
        }

        // Attempt resize_bar(1) (no-op on TB3) and immediately map BAR0, BAR1, BAR3 via `MAP_BAR`
        // BEFORE any `MMIO_READ` / `MMIO_WRITE` is ever issued!
        let _ = client.rpc(REMOTE_CMD_RESIZE_BAR, 1, [0, 0, 0], &[]);
        client.bar_info(0)?;
        client.bar_info(1)?;
        client.bar_info(3)?;

        Ok(client)
    }

    #[inline]
    fn pack_req(cmd: u8, dev_id: u32, bar: u32, args: [u64; 3]) -> [u8; 33] {
        let mut hdr = [0u8; 33];
        hdr[0] = cmd;
        hdr[1..5].copy_from_slice(&dev_id.to_le_bytes());
        hdr[5..9].copy_from_slice(&bar.to_le_bytes());
        hdr[9..17].copy_from_slice(&args[0].to_le_bytes());
        hdr[17..25].copy_from_slice(&args[1].to_le_bytes());
        hdr[25..33].copy_from_slice(&args[2].to_le_bytes());
        hdr
    }

    pub fn rpc(&mut self, cmd: u8, bar: u32, args: [u64; 3], payload: &[u8]) -> Result<(u64, u64), String> {
        let hdr = Self::pack_req(cmd, self.dev_id, bar, args);
        if payload.is_empty() {
            self.sock.write_all(&hdr).map_err(|e| format!("TinyGPU write error: {}", e))?;
        } else {
            let mut combined = Vec::with_capacity(33 + payload.len());
            combined.extend_from_slice(&hdr);
            combined.extend_from_slice(payload);
            self.sock.write_all(&combined).map_err(|e| format!("TinyGPU write error: {}", e))?;
        }

        let mut resp = [0u8; 17];
        self.sock.read_exact(&mut resp).map_err(|e| format!("TinyGPU read error: {}", e))?;
        let status = resp[0];
        let r1 = get_u64(&resp, 1);
        let r2 = get_u64(&resp, 9);
        if status != 0 {
            let err_msg = if r1 > 0 && r1 < 65536 {
                let mut ebuf = vec![0u8; r1 as usize];
                let _ = self.sock.read_exact(&mut ebuf);
                String::from_utf8_lossy(&ebuf).into_owned()
            } else {
                "unknown error".to_string()
            };
            return Err(format!("TinyGPU RPC cmd={} failed: {}", cmd, err_msg));
        }
        Ok((r1, r2))
    }

    /// Fire-and-forget MMIO/BAR write (no 17-byte response on `MMIO_WRITE` in TinyGPU protocol!).
    #[inline]
    pub fn bulk_write(&mut self, bar: u32, offset: u64, data: &[u8]) -> Result<(), String> {
        let hdr = Self::pack_req(REMOTE_CMD_MMIO_WRITE, self.dev_id, bar, [offset, data.len() as u64, 0]);
        if data.len() <= 64 {
            let mut stack_buf = [0u8; 97];
            stack_buf[..33].copy_from_slice(&hdr);
            stack_buf[33..33 + data.len()].copy_from_slice(data);
            self.sock
                .write_all(&stack_buf[..33 + data.len()])
                .map_err(|e| format!("MMIO write failed: {}", e))
        } else {
            self.sock.write_all(&hdr).map_err(|e| e.to_string())?;
            self.sock.write_all(data).map_err(|e| e.to_string())
        }
    }

    #[inline]
    pub fn bulk_read(&mut self, bar: u32, offset: u64, out: &mut [u8]) -> Result<(), String> {
        self.rpc(REMOTE_CMD_MMIO_READ, bar, [offset, out.len() as u64, 0], &[])?;
        self.sock.read_exact(out).map_err(|e| format!("MMIO read payload failed: {}", e))
    }

    #[inline]
    pub fn rreg(&mut self, addr: u32) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.bulk_read(0, addr as u64, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    #[inline]
    pub fn wreg(&mut self, addr: u32, val: u32) -> Result<(), String> {
        self.bulk_write(0, addr as u64, &val.to_le_bytes())
    }

    #[inline]
    pub fn ring_doorbell(&mut self, token: u32) -> Result<(), String> {
        self.bulk_write(0, NV_USERMODE_DOORBELL_BAR0_OFF, &token.to_le_bytes())
    }

    pub fn read_config(&mut self, offset: u64, size: u64) -> Result<u64, String> {
        Ok(self.rpc(REMOTE_CMD_CFG_READ, 0, [offset, size, 0], &[])?.0)
    }

    pub fn write_config(&mut self, offset: u64, val: u64, size: u64) -> Result<(), String> {
        self.rpc(REMOTE_CMD_CFG_WRITE, 0, [offset, size, val], &[])?;
        Ok(())
    }

    pub fn reset_pci(&mut self) -> Result<(), String> {
        self.rpc(REMOTE_CMD_RESET, 0, [0, 0, 0], &[])?;
        self.bar_cache = [(0, 0); 6];
        Ok(())
    }

    pub fn remap_bars(&mut self) -> Result<(), String> {
        self.bar_cache = [(0, 0); 6];
        self.bar_info(0)?;
        self.bar_info(1)?;
        self.bar_info(3)?;
        Ok(())
    }

    pub fn bar_info(&mut self, bar_idx: u32) -> Result<(u64, u64), String> {
        let idx = bar_idx as usize;
        if idx < self.bar_cache.len() && self.bar_cache[idx].1 != 0 {
            return Ok(self.bar_cache[idx]);
        }
        let info = self.rpc(REMOTE_CMD_MAP_BAR, bar_idx, [0, 0, 0], &[])?;
        if idx < self.bar_cache.len() {
            self.bar_cache[idx] = info;
        }
        Ok(info)
    }

    /// Allocate DART-mapped host shared memory via `MAP_SYSMEM_FD` (`SCM_RIGHTS` fd passing).
    pub fn alloc_sysmem(&mut self, size: usize, contiguous: bool) -> Result<SysmemMapping, String> {
        let hdr = Self::pack_req(
            REMOTE_CMD_MAP_SYSMEM_FD,
            self.dev_id,
            0,
            [size as u64, u64::from(contiguous), 0],
        );
        self.sock.write_all(&hdr).map_err(|e| format!("MAP_SYSMEM_FD send failed: {}", e))?;

        let mut resp = [0u8; 17];
        let mut cmsg_space = [0u8; 64]; // ample aligned space for cmsghdr + RawFd
        let mut iov = IoVec {
            iov_base: resp.as_mut_ptr(),
            iov_len: resp.len(),
        };
        let cmsg_len_req = std::mem::size_of::<CMsgHdr>() + std::mem::size_of::<RawFd>();
        let mut msg = MsgHdr {
            msg_name: ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: &mut iov,
            msg_iovlen: 1,
            msg_control: cmsg_space.as_mut_ptr(),
            msg_controllen: cmsg_len_req as u32,
            msg_flags: 0,
        };

        let n = unsafe { recvmsg(self.sock.as_raw_fd(), &mut msg, 0) };
        if n != 17 {
            return Err(format!("MAP_SYSMEM_FD recvmsg returned {} bytes (expected 17)", n));
        }
        if resp[0] != 0 {
            let err_len = get_u64(&resp, 1) as usize;
            let mut ebuf = vec![0u8; err_len.min(4096)];
            let _ = self.sock.read_exact(&mut ebuf);
            return Err(format!("MAP_SYSMEM_FD failed: {}", String::from_utf8_lossy(&ebuf)));
        }
        let mapped_size = get_u64(&resp, 1) as usize;
        if msg.msg_controllen < cmsg_len_req as u32 {
            return Err("MAP_SYSMEM_FD did not return SCM_RIGHTS ancillary fd".into());
        }

        let cmsg = unsafe { &*(cmsg_space.as_ptr() as *const CMsgHdr) };
        if cmsg.cmsg_level != SOL_SOCKET || cmsg.cmsg_type != SCM_RIGHTS {
            return Err(format!(
                "Unexpected cmsg level={} type={}",
                cmsg.cmsg_level, cmsg.cmsg_type
            ));
        }
        let fd_off = std::mem::size_of::<CMsgHdr>();
        let shm_fd = i32::from_le_bytes(cmsg_space[fd_off..fd_off + 4].try_into().unwrap());

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                mapped_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                shm_fd,
                0,
            )
        };
        unsafe { close(shm_fd) };
        if (ptr as isize) == MAP_FAILED_ISIZE || ptr.is_null() {
            return Err(format!("mmap failed for MAP_SYSMEM_FD size={:#x}", mapped_size));
        }

        // Extract (paddr, seg_size) pairs terminated by (0, 0) at start of mapping
        let needed_4k = (size + 0xfff) / 0x1000;
        let mut paddrs_4k = Vec::with_capacity(needed_4k);
        let u64_slice = unsafe { std::slice::from_raw_parts(ptr as *const u64, mapped_size / 8) };
        let mut idx = 0;
        while idx + 1 < u64_slice.len() && paddrs_4k.len() < needed_4k {
            let paddr = u64_slice[idx];
            let seg_sz = u64_slice[idx + 1];
            if seg_sz == 0 {
                break;
            }
            let mut off = 0u64;
            while off < seg_sz && paddrs_4k.len() < needed_4k {
                paddrs_4k.push(paddr + off);
                off += 0x1000;
            }
            idx += 2;
        }
        if paddrs_4k.len() < needed_4k {
            unsafe { munmap(ptr, mapped_size) };
            return Err(format!(
                "MAP_SYSMEM_FD returned only {} 4KB pages, needed {}",
                paddrs_4k.len(),
                needed_4k
            ));
        }

        // Zero the metadata header region we just read so caller gets clean zeroed pages
        unsafe {
            ptr::write_bytes(ptr, 0, ((idx + 2) * 8).min(mapped_size));
        }

        Ok(SysmemMapping {
            ptr,
            mapped_size,
            req_size: size,
            paddrs_4k,
        })
    }
}

impl Drop for TinyGpuClient {
    fn drop(&mut self) {
        unsafe {
            flock(self.lock_file.as_raw_fd(), LOCK_UN);
        }
    }
}
