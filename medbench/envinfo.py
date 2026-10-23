"""Host facts for the manifest and the report: OS, kernel, virtualisation, CPU, RAM/swap, GPU (driver, CUDA, PCIe), disk."""
from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional


def _run(cmd: List[str]) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def collect() -> Dict[str, Any]:
    info: Dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(), "kernel": platform.release(),
                            "cpu": None, "logical_cpus": os.cpu_count(), "ram_gib": None, "swap_gib": None}
    try:
        with open("/etc/os-release") as f:
            info["os"] = next((line.split("=", 1)[1].strip().strip('"') for line in f if line.startswith("PRETTY_NAME=")), None)
    except OSError:
        pass
    virt = _run(["systemd-detect-virt"])
    info["virtualization"] = virt or "none"
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0])
        info["ram_gib"] = round(mem["MemTotal"] / 1024 / 1024, 1)
        info["swap_gib"] = round(mem.get("SwapTotal", 0) / 1024 / 1024, 1)
    except (OSError, StopIteration, KeyError):
        pass
    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if gpu:
        info["gpu"] = gpu.splitlines()[0]
        pcie = _run(["nvidia-smi", "--query-gpu=pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader", "-i", "0"])
        if pcie:
            gen, width = [x.strip() for x in pcie.split(",")[:2]]
            info["gpu_pcie"] = "Gen%s x%s" % (gen, width)
        m = re.search(r"CUDA Version:\s*([0-9.]+)", _run(["nvidia-smi"]) or "")
        if m:
            info["cuda_driver_api"] = m.group(1)
    disk = _run(["lsblk", "-d", "-n", "-e7", "-o", "MODEL,SIZE,ROTA,TYPE"])
    if disk:
        info["disks"] = [" ".join(line.split()) for line in disk.splitlines() if line.strip().endswith("disk")]
    return info
