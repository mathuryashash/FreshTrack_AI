"""Fair CPU latency benchmark across all trained runs.

Per-run latency measured inside evaluate.py is sensitive to whatever else the
machine is doing at that moment. Here every run's model is loaded into one
process and timed in interleaved rounds, so background load affects all models
equally. The result overwrites `cpu_latency_ms_batch1` in each metrics.json.

    python -m src.training.benchmark_latency && python -m src.training.run_experiment --aggregate
"""

import json
import time

import numpy as np
import torch

from src.config import IMAGE_SIZE
from src.models.freshtrack_model import FreshTrackModel
from src.training.train import RUNS_DIR

ROUNDS = 5
ITERS_PER_ROUND = 40


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    runs = sorted(p.parent for p in RUNS_DIR.glob("*_s[0-9]/metrics.json"))
    models = {}
    for run in runs:
        cfg = json.loads((run / "run_config.json").read_text())
        m = FreshTrackModel.load_from_checkpoint(
            cfg["best_checkpoint"], pretrained=False, weights_only=True, map_location="cpu"
        ).eval()
        for _ in range(10):  # warm-up
            m(x)
        models[run] = m

    times = {run: [] for run in runs}
    for _ in range(ROUNDS):
        for run, m in models.items():  # interleaved: same load conditions for all
            for _ in range(ITERS_PER_ROUND):
                t0 = time.perf_counter()
                m(x)
                times[run].append((time.perf_counter() - t0) * 1000)

    for run in runs:
        path = run / "metrics.json"
        metrics = json.loads(path.read_text())
        metrics["cpu_latency_ms_batch1"] = float(np.median(times[run]))
        metrics["cpu_latency_method"] = (
            f"benchmark_latency.py: median of {ROUNDS}x{ITERS_PER_ROUND} interleaved runs, 4 threads"
        )
        path.write_text(json.dumps(metrics, indent=2))
        print(f"{run.name}: {metrics['cpu_latency_ms_batch1']:.2f} ms")


if __name__ == "__main__":
    main()
