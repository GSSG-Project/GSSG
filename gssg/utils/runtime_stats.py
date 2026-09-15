"""Per-frame stage timings -> <save_path>/runtime.json.

One instance lives on Mapping (mapper-internal stages) and the single-process
main loop adds the outer stages to the same object, so runtime.json is the one
machine-readable source for the paper's runtime/memory table.
"""

import json
import os
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class RuntimeStats:
    def __init__(self):
        self.series = defaultdict(list)
        self.extra = {}

    def add(self, stage, ms):
        self.series[stage].append(float(ms))

    @contextmanager
    def timer(self, stage, cuda_sync=False):
        # cuda_sync attributes in-flight GPU work to the stage that launched it;
        # only use at boundaries where the next stage would wait on the GPU anyway.
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.add(stage, (time.perf_counter() - t0) * 1000.0)

    def summary(self):
        out = {}
        for stage, vals in self.series.items():
            s = sorted(vals)
            out[stage] = {
                "mean_ms": sum(s) / len(s),
                "median_ms": statistics.median(s),
                "p90_ms": s[min(len(s) - 1, int(0.9 * len(s)))],
                "total_s": sum(s) / 1000.0,
                "n": len(s),
            }
        return out

    def write(self, save_path):
        payload = {"stages": self.summary(), **self.extra}
        if torch.cuda.is_available():
            payload["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
            payload["device"] = torch.cuda.get_device_name(0)
        path = os.path.join(save_path, "runtime.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        self.write_series(save_path)
        return path

    def write_series(self, save_path):
        """Raw per-frame series for the Fig. 4 scaling figure (kept out of
        runtime.json so the summary stays small). Safe to call mid-run."""
        payload = {"stages_raw": {k: list(v) for k, v in self.series.items()}}
        if "scaling" in self.extra:
            payload["scaling"] = self.extra["scaling"]
        spath = os.path.join(save_path, "runtime_series.json")
        os.makedirs(save_path, exist_ok=True)
        with open(spath + ".tmp", "w") as f:
            json.dump(payload, f)
        os.replace(spath + ".tmp", spath)
        return spath
