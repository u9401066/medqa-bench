"""Run a model over benchmark items: streamed OpenAI-compatible chat completions, bounded concurrency, resumable.

Per item it records the parsed answer and timing (time to first token, wall latency, completion tokens, decode speed) in
``predictions.jsonl`` and the full output (reasoning + answer) in ``raw.jsonl``. ``manifest.json`` holds everything needed to
reproduce the run (config, versions, server properties, dataset hashes, host).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from . import __version__
from .datasets import SOURCES, load, sha256
from .monitor import Monitor
from .prompts import PROMPT_VERSION, build_prompt, parse

TIMING_KEYS = ("prompt_n", "prompt_ms", "cache_n", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")


def item_seed(item_id: str) -> int:
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % (2 ** 31)


def select(items: List[Dict[str, Any]], sample: Optional[int]) -> List[Dict[str, Any]]:
    """Deterministic, evenly spaced subset (pilot runs); the full set when sample is None."""
    if not sample or sample >= len(items):
        return items
    return [items[int(i * len(items) / sample)] for i in range(sample)]


async def ask(client: httpx.AsyncClient, cfg: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": cfg.get("model", "model"), "messages": [{"role": "user", "content": build_prompt(item)}],
                            "stream": True, "stream_options": {"include_usage": True}, "seed": item_seed(item["id"])}
    body.update(cfg.get("request") or {})
    t0 = time.time()
    first: Optional[float] = None
    content, reasoning = [], []
    usage: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}
    finish = None
    async with client.stream("POST", "/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            raise httpx.HTTPStatusError("HTTP %d: %s" % (r.status_code, (await r.aread())[:300]), request=r.request, response=r)
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.startswith("data: [DONE]"):
                continue
            d = json.loads(line[6:])
            if d.get("error"):
                raise RuntimeError(str(d["error"])[:300])
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                c, rc = delta.get("content"), delta.get("reasoning_content") or delta.get("reasoning")
                if (c or rc) and first is None:
                    first = time.time()
                if c:
                    content.append(c)
                if rc:
                    reasoning.append(rc)
                finish = ch.get("finish_reason") or finish
            usage = d.get("usage") or usage
            timings = d.get("timings") or timings
    t1 = time.time()
    text = "".join(content)
    n_out = int(usage.get("completion_tokens") or 0)
    rec = {"t_start": round(t0, 3), "latency_s": round(t1 - t0, 3), "ttft_s": round(first - t0, 3) if first else None,
           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n_out, "finish_reason": finish,
           "decode_tps": round((n_out - 1) / (t1 - first), 2) if first and n_out > 1 and t1 > first else None,
           "content_chars": len(text), "reasoning_chars": len("".join(reasoning))}
    if timings:
        rec["timings"] = {k: timings.get(k) for k in TIMING_KEYS if k in timings}
    return {"rec": rec, "content": text, "reasoning": "".join(reasoning)}


def _git_rev() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _host() -> Dict[str, Any]:
    info: Dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(), "cpu": None, "ram_gib": None}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
        with open("/proc/meminfo") as f:
            info["ram_gib"] = round(int(f.readline().split()[1]) / 1024 / 1024, 1)
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        info["gpu"] = out
    except Exception:  # noqa: BLE001
        pass
    return info


async def _server_info(client: httpx.AsyncClient, paths: List[str]) -> Dict[str, Any]:
    out = {}
    for p in paths:
        try:
            r = await client.get(p, timeout=10)
            data = r.json()
            if isinstance(data, dict):
                data.pop("chat_template", None)  # large; the template is part of the model file
            out[p] = data
        except Exception as e:  # noqa: BLE001
            out[p] = {"error": type(e).__name__}
    return out


async def run(cfg: Dict[str, Any], data_dir: Path, datasets: List[str], out: Path, sample: Optional[int] = None,
              retries: int = 2, baseline_s: float = 15.0) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    pred_path, raw_path = out / "predictions.jsonl", out / "raw.jsonl"
    done: Dict[str, Dict[str, Any]] = {}
    if pred_path.exists():
        for line in open(pred_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done[r["id"]] = r
    items = [it for name in datasets for it in select(load(data_dir, name), sample)]
    todo = [it for it in items if it["id"] not in done]
    state = {"inflight": 0, "done": len(items) - len(todo)}
    timeout = httpx.Timeout(cfg.get("timeout_s", 1800), connect=30)
    async with httpx.AsyncClient(base_url=cfg["base_url"].rstrip("/"), timeout=timeout, headers=cfg.get("headers") or {}) as client:
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update({"medbench_version": __version__, "git_rev": _git_rev(), "prompt_version": PROMPT_VERSION, "config": cfg,
                         "datasets": {n: {"n_items": len(select(load(data_dir, n), sample)), "sample": sample, "revision": SOURCES[n].revision,
                                          "sha256": SOURCES[n].sha256, "items_sha256": sha256(data_dir / ("%s.jsonl" % n))} for n in datasets},
                         "host": _host(), "server": await _server_info(client, cfg.get("server_info") or ["/v1/models"])})
        slots = cfg.get("slots_path", "/slots")
        mon = Monitor(out / "resources.csv", cfg.get("processes") or [], state=lambda: dict(state),
                      slots_url=cfg["base_url"].rstrip("/") + slots if slots else None).start()
        # Idle samples before the first request give the power baseline (energy above idle).
        session: Dict[str, Any] = {"baseline_start": time.time(), "todo": len(todo)}
        if todo and baseline_s > 0:
            await asyncio.sleep(baseline_s)
        session["start"] = time.time()
        manifest.setdefault("sessions", []).append(session)
        manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        sem = asyncio.Semaphore(int(cfg.get("concurrency", 1)))
        lock = asyncio.Lock()
        t_start = time.time()

        async def one(item: Dict[str, Any]) -> None:
            async with sem:
                state["inflight"] += 1
                res, err = None, None
                for attempt in range(retries + 1):
                    try:
                        res = await ask(client, cfg, item)
                        err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        err = "%s: %s" % (type(e).__name__, str(e)[:300])
                        await asyncio.sleep(5 * (attempt + 1))
                state["inflight"] -= 1
                rec: Dict[str, Any] = {"id": item["id"], "dataset": item["dataset"], "gold": item["answer"]}
                if res is None:
                    rec.update({"pred": None, "correct": False, "error": err})
                else:
                    pred = parse(item, res["content"])
                    rec.update({"pred": pred, "correct": pred == item["answer"], **res["rec"]})
                async with lock:
                    with open(pred_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if res is not None:
                        with open(raw_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"id": item["id"], "content": res["content"], "reasoning": res["reasoning"]}, ensure_ascii=False) + "\n")
                    state["done"] += 1
                    n = state["done"]
                    if n % 25 == 0 or n == len(items):
                        el = time.time() - t_start
                        print("[%s] %d/%d done, %.0f s elapsed" % (cfg["name"], n, len(items), el), flush=True)

        # SIGINT/SIGTERM stop the run cleanly (in-flight items are dropped and redone on resume). Explicit handlers, because a
        # background shell starts the process with SIGINT ignored.
        work = asyncio.ensure_future(asyncio.gather(*[one(it) for it in todo]))
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, work.cancel)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            await work
        except asyncio.CancelledError:
            print("[%s] stopped by signal; %d/%d done, resume with the same command" % (cfg["name"], state["done"], len(items)), flush=True)
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
            mon.stop()
            manifest["sessions"][-1]["end"] = time.time()
            manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return {"items": len(items), "ran": len(todo)}
