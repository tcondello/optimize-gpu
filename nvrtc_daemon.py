#!/usr/bin/env python3
"""
Zero-Mount Persistent CUDA Compiler Daemon (NVRTC + /dev/shm NVCC fallback).

Eliminates two major macOS + TinyGPU bottlenecks:
1. Docker Desktop VirtioFS crashes (`service fs failed: injecting event blocked for 60s`)
   caused by mounting `-v $HOME:$HOME -v /var/folders:/var/folders` in `nvccshim`.
   This container uses ZERO host volume mounts (`-p 127.0.0.1:19577:19577` only).
2. 550-900 ms per-kernel compile overhead from `docker inspect` + `docker exec` + `nvcc` fork.
   Uses in-memory `libnvrtc.so.12` inside a persistent multi-threaded Python server (~8-15 ms/kernel),
   with automatic fallback to `/dev/shm` RAM-disk `nvcc` when full CUDA headers (`mma.sync`, etc.) require it.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = int(os.getenv("TINYGPU_NVRTC_PORT", "19577"))
CONTAINER_NAME = "cuda-nvcc-fast-daemon"
IMAGE_NAME = "cuda-nvcc:12.8-fast"

CONTAINER_SERVER_SCRIPT = r'''
import ctypes, hashlib, json, os, socketserver, struct, subprocess, tempfile, threading

# Load libnvrtc for ultra-fast in-memory CUDA -> CUBIN compilation
nvrtc = ctypes.CDLL("libnvrtc.so.12")
nvrtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
nvrtc.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

_cache = {}
_cache_lock = threading.Lock()

def compile_nvrtc(src: bytes, arch: str, mode: str, extra_opts: list[str]) -> bytes:
    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"kernel.cu", 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {rc}")
    try:
        opts = [f"--gpu-architecture={arch}".encode(), b"-I/usr/local/cuda/include", b"-I/usr/include"]
        for o in extra_opts:
            if o:
                opts.append(o.encode())
        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
        if rc != 0:
            log_sz = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            buf = ctypes.create_string_buffer(log_sz.value)
            nvrtc.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError(f"nvrtcCompileProgram ({rc}): {buf.value.decode(errors='replace')}")
        out_sz = ctypes.c_size_t()
        if mode == "ptx":
            if nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(out_sz)) != 0:
                raise RuntimeError("nvrtcGetPTXSize failed")
            out = ctypes.create_string_buffer(out_sz.value)
            nvrtc.nvrtcGetPTX(prog, out)
        else:
            if nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(out_sz)) != 0:
                raise RuntimeError("nvrtcGetCUBINSize failed")
            out = ctypes.create_string_buffer(out_sz.value)
            nvrtc.nvrtcGetCUBIN(prog, out)
        return ctypes.string_at(out, out_sz.value)
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

def compile_nvcc_shm(src: bytes, arch: str, mode: str, extra_opts: list[str]) -> bytes:
    suffix = ".ptx" if mode == "ptx" else ".cubin"
    flag = "-ptx" if mode == "ptx" else "-cubin"
    with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
        cu_path = os.path.join(td, "kern.cu")
        out_path = os.path.join(td, "kern" + suffix)
        with open(cu_path, "wb") as f:
            f.write(src)
        cmd = ["/usr/local/cuda/bin/nvcc", f"-arch={arch}", flag, "-O3", "--use_fast_math", "-o", out_path, cu_path] + [o for o in extra_opts if o]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace") or proc.stdout.decode(errors="replace"))
        with open(out_path, "rb") as f:
            return f.read()

class Handler(socketserver.BaseRequestHandler):
    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.request.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("EOF")
            buf.extend(chunk)
        return bytes(buf)

    def handle(self):
        while True:
            try:
                hdr_len_raw = self.request.recv(4)
                if not hdr_len_raw:
                    return
                if len(hdr_len_raw) < 4:
                    hdr_len_raw += self._recv_exact(4 - len(hdr_len_raw))
                hdr_len, src_len = struct.unpack("<II", hdr_len_raw + self._recv_exact(4))
                meta = json.loads(self._recv_exact(hdr_len).decode())
                src = self._recv_exact(src_len)
            except ConnectionError:
                return

            cmd = meta.get("cmd", "compile")
            if cmd == "ping":
                self.request.sendall(struct.pack("<BI", 0, 2) + b"OK")
                continue

            arch = meta.get("arch", "sm_86")
            mode = meta.get("mode", "cubin")
            backend = meta.get("backend", "auto")
            extra_opts = meta.get("extra_opts", [])
            key = hashlib.sha256(json.dumps([arch, mode, backend, extra_opts]).encode() + b"\0" + src).digest()

            with _cache_lock:
                cached = _cache.get(key)
            if cached is not None:
                self.request.sendall(struct.pack("<BI", 0, len(cached)) + cached)
                continue

            try:
                if backend == "nvcc":
                    out = compile_nvcc_shm(src, arch, mode, extra_opts)
                else:
                    try:
                        out = compile_nvrtc(src, arch, mode, extra_opts)
                    except Exception:
                        if backend == "nvrtc":
                            raise
                        out = compile_nvcc_shm(src, arch, mode, extra_opts)
                with _cache_lock:
                    _cache[key] = out
                self.request.sendall(struct.pack("<BI", 0, len(out)) + out)
            except Exception as e:
                err = str(e).encode()
                self.request.sendall(struct.pack("<BI", 1, len(err)) + err)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    with ThreadedTCPServer(("0.0.0.0", 19577), Handler) as srv:
        srv.serve_forever()
'''


def _ping_daemon(timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((DAEMON_HOST, DAEMON_PORT), timeout=timeout) as sock:
            hdr = b'{"cmd":"ping"}'
            sock.sendall(struct.pack("<II", len(hdr), 0) + hdr)
            resp = sock.recv(7)
            return len(resp) == 7 and resp[:1] == b"\x00" and resp[5:] == b"OK"
    except OSError:
        return False


def ensure_compiler_daemon() -> None:
    """Ensure the zero-mount NVRTC/NVCC container daemon is up and responsive."""
    if _ping_daemon():
        return

    # Ensure Docker Desktop is up
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        subprocess.run(["open", "-g", "-a", "Docker"], check=False)
        for _ in range(45):
            if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Docker daemon is not running and could not be started.")

    # Remove legacy VirtioFS-mounting container if present so it cannot stall Docker Desktop
    subprocess.run(["docker", "rm", "-f", "cuda-nvcc-persistent"], capture_output=True)

    # Build cuda-nvcc:12.8-fast if missing
    if subprocess.run(["docker", "image", "inspect", IMAGE_NAME], capture_output=True).returncode != 0:
        dockerfile = (
            "FROM cuda-nvcc:12.8\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends cuda-nvrtc-12-8 python3 "
            "&& rm -rf /var/lib/apt/lists/*\n"
        )
        subprocess.run(
            ["docker", "build", "--platform=linux/arm64", "-t", IMAGE_NAME, "-"],
            input=dockerfile.encode(),
            check=True,
        )

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d",
            "--platform=linux/arm64",
            "--restart", "unless-stopped",
            "--name", CONTAINER_NAME,
            "-p", f"{DAEMON_HOST}:{DAEMON_PORT}:19577",
            IMAGE_NAME,
            "python3", "-u", "-c", CONTAINER_SERVER_SCRIPT,
        ],
        check=True,
        capture_output=True,
    )

    for _ in range(50):
        if _ping_daemon(timeout=0.5):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Failed to connect to {CONTAINER_NAME} on {DAEMON_HOST}:{DAEMON_PORT}")


def daemon_compile(
    src: str | bytes,
    arch: str = "sm_86",
    mode: str = "cubin",
    backend: str = "auto",
    extra_opts: list[str] | None = None,
) -> bytes:
    """Compile CUDA C++ source to SASS .cubin or PTX over localhost TCP in ~8-15 ms."""
    src_bytes = src.encode() if isinstance(src, str) else src
    hdr = json.dumps({
        "cmd": "compile",
        "arch": arch,
        "mode": mode,
        "backend": backend,
        "extra_opts": extra_opts or [],
    }).encode()

    ensure_compiler_daemon()
    with socket.create_connection((DAEMON_HOST, DAEMON_PORT), timeout=30.0) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(struct.pack("<II", len(hdr), len(src_bytes)) + hdr + src_bytes)
        resp_hdr = b""
        while len(resp_hdr) < 5:
            chunk = sock.recv(5 - len(resp_hdr))
            if not chunk:
                raise RuntimeError("Compiler daemon closed connection")
            resp_hdr += chunk
        status, length = struct.unpack("<BI", resp_hdr)
        payload = bytearray()
        while len(payload) < length:
            chunk = sock.recv(length - len(payload))
            if not chunk:
                raise RuntimeError("Compiler daemon truncated payload")
            payload.extend(chunk)
        if status != 0:
            raise RuntimeError(f"CUDA compile failed: {payload.decode(errors='replace')}")
        return bytes(payload)


def install_fast_tinygrad_compiler() -> None:
    """Monkeypatch tinygrad's NVCCCompiler so ALL tinygrad & BEAM kernels compile via NVRTC daemon (~15ms)."""
    from tinygrad.device import CompileError
    from tinygrad.runtime.support import compiler_cuda

    def _fast_compile(self, src: str) -> bytes:
        mode = "ptx" if self.ptx else "cubin"
        try:
            return daemon_compile(src, arch=self.arch, mode=mode, backend="auto", extra_opts=list(self.extra_options))
        except Exception as e:
            raise CompileError(str(e)) from e

    compiler_cuda.NVCCCompiler.compile = _fast_compile


def install_nv_thunderbolt_membar_opt() -> None:
    """
    Eliminate 350+ per-kernel PCIe/Thunderbolt SYSMEMBAR round-trips per token!

    Stock `tinygrad/runtime/ops_nv.py` sets `cwd_membar_type = L1_SYSMEMBAR (1)` on EVERY QMD,
    forcing a PCIe root-complex flush across Thunderbolt (~3-4 us) after every kernel in HCQGraph.
    Using `L1_MEMBAR (3)` orders all GPU VRAM reads/writes internally in L2 (~0.1 us), while
    enabling `FE_SYSMEMBAR (1)` on `release{i}_membar_type` only when signaling the host CPU.
    """
    import tinygrad.runtime.ops_nv as ops_nv

    if getattr(ops_nv, "_thunderbolt_membar_patched", False):
        return

    orig_nvprog_init = ops_nv.NVProgram.__init__

    def _fast_nvprog_init(self, dev, name, lib, **kwargs):
        orig_nvprog_init(self, dev, name, lib, **kwargs)
        if self.qmd.ver < 4:
            self.qmd.write(cwd_membar_type=ops_nv.nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_MEMBAR)

    orig_signal = ops_nv.NVComputeQueue.signal

    def _fast_signal(self, signal, value=0):
        if self.active_qmd is not None and self.active_qmd.ver < 4:
            for i in range(2):
                if self.active_qmd.read(f"release{i}_enable") == 0:
                    self.active_qmd.write(**{f"release{i}_membar_type": 1})
                    break
        return orig_signal(self, signal, value)

    ops_nv.NVProgram.__init__ = _fast_nvprog_init
    ops_nv.NVComputeQueue.signal = _fast_signal
    ops_nv._thunderbolt_membar_patched = True


if __name__ == "__main__":
    t0 = time.perf_counter()
    ensure_compiler_daemon()
    t1 = time.perf_counter()
    test_cu = 'extern "C" __global__ void test_k(float* out) { out[threadIdx.x] = 1.0f; }'
    cubin1 = daemon_compile(test_cu + " // 1", arch="sm_86")
    t2 = time.perf_counter()
    cubin2 = daemon_compile(test_cu + " // 2", arch="sm_86")
    t3 = time.perf_counter()
    print(f"Daemon ready in {(t1-t0)*1e3:.1f} ms | Cold NVRTC compile: {(t2-t1)*1e3:.2f} ms ({len(cubin1)} B) | Warm NVRTC compile: {(t3-t2)*1e3:.2f} ms ({len(cubin2)} B)")
