#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.adapters.csfm import (
    CSFM_CHANNELS,
    CSFM_DEPTH,
    prepare_inputs,
    try_load_model,
)
from benchmark_v1.adapters.ecg_jepa import DEFAULT_SPLIT_CSV, read_split_ids
from benchmark_v1.config import ROOT


DEFAULT_OUT_DIR = ROOT / "results" / "analysis" / "csfm_cu118_commons"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CSFM continuation from intermediate transformer layers.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def build_transformer_input(model, source, channel):
    import torch
    from einops import rearrange, repeat

    ts = model.ts_to_patch_embedding(source)
    b, c, n, _ = ts.shape
    ts_channel_emb = repeat(model.ts_channel_type_embedding[:, channel], "1 c d -> b c n d", n=n, b=b)
    ts = ts + ts_channel_emb
    ts_position_emb = repeat(model.ts_pos_embedding, "1 n d -> b c n d", c=c, b=b)
    ts = ts + ts_position_emb
    ts = rearrange(ts, "b c n d -> b (c n) d")
    cls_tokens = repeat(model.cls_token, "1 1 d -> b 1 d", b=b)
    x = torch.cat((cls_tokens, ts), dim=1)
    return model.dropout(x)


def pool_output(model, x):
    x = x.mean(dim=1) if model.pool == "mean" else x[:, 0]
    x = model.to_latent(x)
    return model.mlp_head(x)


def continue_from_post_block(model, x, layer_idx: int, mask=None):
    for attn, ff in model.transformer.layers[layer_idx + 1 :]:
        x = attn(x, mask=mask) + x
        x = ff(x) + x
    x = model.transformer.norm(x)
    return pool_output(model, x)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ecg_ids = read_split_ids(args.split, args.limit, DEFAULT_SPLIT_CSV, offset=args.offset)
    batch, meta = prepare_inputs(ecg_ids)
    if batch is None:
        raise RuntimeError(f"could not prepare inputs: {meta}")

    import numpy as np
    import torch

    model, status = try_load_model(device=args.device)
    if model is None:
        raise RuntimeError(status)

    source = torch.as_tensor(batch, dtype=torch.float32, device=args.device)
    channel = torch.as_tensor(CSFM_CHANNELS, dtype=torch.long, device=args.device)
    layer_states = {}
    manual_x = build_transformer_input(model, source, channel)
    with torch.no_grad():
        full = model(source, channel, task="cls")
        x = manual_x
        for layer_idx, (attn, ff) in enumerate(model.transformer.layers):
            x = attn(x, mask=None) + x
            x = ff(x) + x
            layer_states[layer_idx] = x.clone()
        manual_full = pool_output(model, model.transformer.norm(x))
        rows = []
        for layer_idx in range(CSFM_DEPTH):
            continued = continue_from_post_block(model, layer_states[layer_idx], layer_idx)
            diff = (continued - full).detach().abs().max().item()
            manual_diff = (continued - manual_full).detach().abs().max().item()
            rows.append(
                {
                    "layer": layer_idx,
                    "max_abs_diff_vs_model_forward": diff,
                    "max_abs_diff_vs_manual_forward": manual_diff,
                }
            )

    report = {
        "model_status": status,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "ecg_ids": ecg_ids,
        "input_shape": list(batch.shape),
        "full_shape": list(full.detach().cpu().numpy().shape),
        "manual_full_max_abs_diff": float((manual_full - full).detach().abs().max().item()),
        "layer_continuation": rows,
        "passed": all(row["max_abs_diff_vs_model_forward"] < 1e-5 for row in rows),
    }
    out_path = args.out_dir / "csfm_continuation_check.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
