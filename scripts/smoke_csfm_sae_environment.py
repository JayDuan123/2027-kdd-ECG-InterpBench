#!/usr/bin/env python
"""Identity-forward smoke test for CSFM SAE environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.sae_extension.csfm_environment import CSFMSAEEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", default="qrs_duration")
    parser.add_argument("--task", default="ptbxl_cd")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-test-shards", type=int, default=1)
    args = parser.parse_args()

    env = CSFMSAEEnvironment(device=args.device, max_test_shards=args.max_test_shards)
    env.set_active_task(args.task)
    train = env.load_activations("CSFM", args.layer, "train")
    test = env.load_activations("CSFM", args.layer, "test")
    U = env.load_leace_subspace("CSFM", args.concept, args.task, args.layer)
    cav = env.load_cav("CSFM", args.concept, args.layer)
    auroc = env.forward_with_patch("CSFM", args.layer, "test", lambda acts: acts)
    report = {
        "concept": args.concept,
        "task": args.task,
        "layer": args.layer,
        "device": args.device,
        "max_test_shards": args.max_test_shards,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "leace_u_shape": list(U.shape),
        "cav_shape": list(cav.shape),
        "identity_auroc": auroc,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
