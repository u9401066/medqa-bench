"""Resource sampler: GPU (nvidia-smi), system RAM, CPU and the RSS of the model-server processes, once per second, to CSV."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

FIELDS = ["t", "gpu_mem_used_mib", "gpu_util_pct", "gpu_power_w", "gpu_temp_c", "ram_used_mib", "server_rss_mib", "cpu_pct",
          "inflight", "server_busy", "done"]


def _gpu() -> Dict[str, float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                              "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=5).stdout
        mem, util, power, temp = [float(x) for x in out.strip().split(",")[:4]]
        return {"gpu_mem_used_mib": mem, "gpu_util_pct": util, "gpu_power_w": power, "gpu_temp_c": temp}
    except Exception:  # noqa: BLE001
        return {}


def _ram_used_mib() -> float:
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 1024.0


def _cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return idle, sum(v)


def server_pids(patterns: List[str]) -> List[int]:
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open("/proc/%s/cmdline" % d, "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if any(p in cmd for p in patterns) and "medbench" not in cmd:
            pids.append(int(d))
    return pids


def _busy_slots(url: str) -> Optional[int]:
    """Slots processing a request (llama-server and Strata /slots); with our own in-flight count this shows other users' load."""
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        slots = data.get("slots", []) if isinstance(data, dict) else data
        return sum(1 for x in slots if x.get("is_processing"))
    except Exception:  # noqa: BLE001
        return None


def _rss_mib(pids: List[int]) -> float:
    total = 0
    for p in pids:
        try:
            with open("/proc/%d/status" % p) as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except OSError:
            pass
    return total / 1024.0


class Monitor:
    def __init__(self, path: Path, patterns: List[str], interval: float = 1.0, state: Optional[Callable[[], Dict[str, int]]] = None,
                 slots_url: Optional[str] = None):
        self.path, self.patterns, self.interval, self.slots_url = path, patterns, interval, slots_url
        self.state = state or (lambda: {})
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            idle0, tot0 = _cpu_times()
            pids, pids_at = server_pids(self.patterns), time.time()
            while not self._stop.is_set():
                t = time.time()
                if t - pids_at > 30:
                    pids, pids_at = server_pids(self.patterns), t
                idle1, tot1 = _cpu_times()
                cpu = 100.0 * (1 - (idle1 - idle0) / max(1, tot1 - tot0))
                idle0, tot0 = idle1, tot1
                row = {"t": round(t, 2), "ram_used_mib": round(_ram_used_mib(), 1), "server_rss_mib": round(_rss_mib(pids), 1),
                       "cpu_pct": round(cpu, 1), **_gpu(), **self.state()}
                if self.slots_url:
                    row["server_busy"] = _busy_slots(self.slots_url)
                w.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                self._stop.wait(max(0.0, self.interval - (time.time() - t)))

    def start(self) -> "Monitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(10)
