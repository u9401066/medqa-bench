import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from medbench.cli import main
from medbench.score import compare, macro_f1, mcnemar_exact, score_run, wilson


def test_wilson_known_values():
    lo, hi = wilson(81, 100)
    assert math.isclose(lo, 0.7222, abs_tol=1e-3) and math.isclose(hi, 0.8749, abs_tol=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)
    assert math.isclose(wilson(10, 10)[1], 1.0)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(1, 9), 2 * 11 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def test_macro_f1():
    assert macro_f1(["yes", "no", "maybe"], ["yes", "no", "maybe"], ["yes", "no", "maybe"]) == 1.0
    assert math.isclose(macro_f1(["yes", "yes", "no"], ["yes", None, "no"], ["yes", "no"]), (2 / 3 + 1) / 2)


ITEMS = {
    "medqa_usmle": [{"id": "medqa_usmle-%04d" % i, "dataset": "medqa_usmle", "lang": "en", "kind": "mcq", "question": "Q%d?" % i,
                     "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "context": None, "answer": "ABCD"[i % 4],
                     "meta": {"meta_info": "step1" if i < 3 else "step2&3"}} for i in range(6)],
    "pubmedqa": [{"id": "pubmedqa-%04d" % i, "dataset": "pubmedqa", "lang": "en", "kind": "yesno", "question": "R%d?" % i,
                  "options": None, "context": "BACKGROUND: x", "answer": ["yes", "no", "maybe"][i % 3], "meta": {"pubid": i}} for i in range(3)],
}


class FakeOpenAI(BaseHTTPRequestHandler):
    """Streams a reasoning chunk and an answer; answers Q0..Q3 correctly, Q4 wrongly, Q5 without an answer line."""

    answers = {"Q0?": "A", "Q1?": "B", "Q2?": "C", "Q3?": "D", "Q4?": "B", "Q5?": None, "R0?": "yes", "R1?": "maybe", "R2?": "maybe"}

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        key = next(k for k in self.answers if ("\n\n%s\n\n" % k) in prompt or ("Question: %s" % k) in prompt)
        ans = self.answers[key]
        text = "I am not sure." if ans is None else "Reasoned.\nAnswer: %s" % ans
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{"choices": [{"delta": {"reasoning_content": "thinking about %s" % key}}]},
                  {"choices": [{"delta": {"content": text}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"prompt_n": 50, "predicted_n": 20}},
                  {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20}}]
        for c in chunks:
            self.wfile.write(("data: %s\n\n" % json.dumps(c)).encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data": [{"id": "fake"}]}')

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def _data(tmp: Path) -> Path:
    d = tmp / "data"
    d.mkdir()
    for name, items in ITEMS.items():
        (d / ("%s.jsonl" % name)).write_text("".join(json.dumps(it) + "\n" for it in items))
    return d


def _cfg(tmp: Path, url: str, name: str) -> Path:
    p = tmp / ("%s.json" % name)
    p.write_text(json.dumps({"name": name, "label": name.upper(), "base_url": url, "concurrency": 2, "request": {"temperature": 0},
                             "server_info": ["/v1/models"], "system": {"model": "fake"}}))
    return p


def test_end_to_end_run_score_report(tmp_path, fake_server):
    data = _data(tmp_path)
    runs = []
    for name in ("m1", "m2"):
        out = tmp_path / "results" / name
        rc = main(["--data", str(data), "run", "--config", str(_cfg(tmp_path, fake_server, name)), "--datasets", "medqa_usmle,pubmedqa",
                   "--out", str(out), "--baseline-s", "0"])
        assert rc == 0
        runs.append(out)
    preds = [json.loads(line) for line in (runs[0] / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 9 and all(p["completion_tokens"] == 20 and p["timings"]["prompt_n"] == 50 for p in preds)
    assert len((runs[0] / "raw.jsonl").read_text().splitlines()) == 9

    # Resume: nothing left to do, nothing appended.
    rc = main(["--data", str(data), "run", "--config", str(_cfg(tmp_path, fake_server, "m1")), "--datasets", "medqa_usmle,pubmedqa",
               "--out", str(runs[0]), "--baseline-s", "0"])
    assert rc == 0 and len((runs[0] / "predictions.jsonl").read_text().splitlines()) == 9

    s = score_run(runs[0], data)
    us, pq = s["datasets"]["medqa_usmle"], s["datasets"]["pubmedqa"]
    assert (us["correct"], us["n"], us["no_answer"]) == (4, 6, 1)
    assert us["subsets"]["step1"]["correct"] == 3 and us["subsets"]["step2&3"]["correct"] == 1
    assert (pq["correct"], pq["n"]) == (2, 3)
    assert s["overall"]["correct"] == 6 and s["overall"]["n"] == 9

    c = compare(runs[0], runs[1])
    assert c["ALL"]["only_a"] == 0 and c["ALL"]["only_b"] == 0 and c["ALL"]["mcnemar_p"] == 1.0

    rep = tmp_path / "reports" / "REPORT.md"
    notes = tmp_path / "notes.md"
    notes.write_text("## 6. 討論\n\n測試。")
    assert main(["--data", str(data), "report", str(runs[0]), str(runs[1]), "--out", str(rep), "--notes", str(notes)]) == 0
    text = rep.read_text()
    assert "## 2. 正確率" in text and "M1" in text and "## 6. 討論" in text and "McNemar" in text
    assert (tmp_path / "reports" / "REPORT.json").exists()


def test_failed_items_are_retried_on_resume(tmp_path, fake_server):
    data = _data(tmp_path)
    out = tmp_path / "results" / "m"
    out.mkdir(parents=True)
    bad = {"id": "pubmedqa-0000", "dataset": "pubmedqa", "gold": "yes", "pred": None, "correct": False, "error": "ConnectError"}
    (out / "predictions.jsonl").write_text(json.dumps(bad) + "\n")
    assert main(["--data", str(data), "run", "--config", str(_cfg(tmp_path, fake_server, "m")), "--datasets", "pubmedqa",
                 "--out", str(out), "--baseline-s", "0"]) == 0
    s = score_run(out, data)
    assert s["datasets"]["pubmedqa"]["errors"] == 0 and s["datasets"]["pubmedqa"]["correct"] == 2


class SlowOpenAI(FakeOpenAI):
    def do_POST(self):  # noqa: N802
        import time

        time.sleep(0.5)
        super().do_POST()


def test_sigint_stops_cleanly_even_when_ignored(tmp_path):
    import signal
    import subprocess
    import sys
    import time

    srv = ThreadingHTTPServer(("127.0.0.1", 0), SlowOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        data = _data(tmp_path)
        cfg = _cfg(tmp_path, "http://127.0.0.1:%d" % srv.server_address[1], "slow")
        cfg.write_text(json.dumps({**json.loads(cfg.read_text()), "concurrency": 1}))
        out = tmp_path / "results" / "slow"
        # a background shell starts its children with SIGINT ignored; the runner must still stop on it
        p = subprocess.Popen([sys.executable, "-m", "medbench.cli", "--data", str(data), "run", "--config", str(cfg), "--datasets",
                              "medqa_usmle", "--out", str(out), "--baseline-s", "0"], preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(2.0)
        p.send_signal(signal.SIGINT)
        stdout, _ = p.communicate(timeout=30)
        assert p.returncode == 0, stdout
        assert "stopped by signal" in stdout
        m = json.loads((out / "manifest.json").read_text())
        assert "end" in m["sessions"][-1]
        n = len((out / "predictions.jsonl").read_text().splitlines())
        assert 1 <= n < 6
    finally:
        srv.shutdown()


def test_session_without_end_is_closed_at_last_item():
    from medbench.score import _close_sessions

    ok = [{"t_start": 10.0, "latency_s": 5.0}, {"t_start": 12.0, "latency_s": 9.0}, {"t_start": 105.0, "latency_s": 1.0}]
    out = _close_sessions([{"baseline_start": 0, "start": 5.0}, {"baseline_start": 90.0, "start": 100.0, "end": 110.0}], ok)
    assert out[0]["end"] == 21.0 and out[0]["end_inferred"] and out[1]["end"] == 110.0
