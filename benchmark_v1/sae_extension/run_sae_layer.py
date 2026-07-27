"""Orchestrator skeleton for the experimental SAE extension.

This script is intentionally not wired to the main benchmark. It requires a
real Environment implementation and will fail with StubEnvironment until the
LEACE U_r basis and CAV artifacts are persisted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .environment import StubEnvironment
from .metrics import (
    FiducialDecoder,
    clamp_reconstruct,
    concept_neutral_centroid,
    coupling_prior_wbi_test,
    decomposability,
    geometric_agreement,
    patient_bootstrap_selectivity,
    run_endpoint_selectivity,
    run_selectivity_sweep,
    select_concept_features,
    select_target_features,
    sparsity_n90,
)
from .train_sae import SAEFit, select_operating_point, sweep_operating_points


PILOT_CELLS = {
    ("CSFM", "st_amp_global", "mi_ischemia"),
    ("CSFM", "qrs_duration", "ptbxl_cd"),
    ("CSFM", "qrst_angle", "ptbxl_cd"),
    ("CSFM", "p_found", "af_rhythm"),
    ("HuBERT-ECG", "qrst_angle", "mi_ischemia"),
    ("HuBERT-ECG", "q_amp_precordial", "ptbxl_mi"),
    ("HuBERT-ECG", "hr_atrial", "af_rhythm"),
}


def parse_candidate(candidate: str) -> tuple[str, str, int]:
    concept, rest = candidate.split("->", 1)
    task, layer = rest.rsplit("@L", 1)
    return concept, task, int(layer)


def parse_int_grid(spec: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("grid specification must contain at least one integer")
    return values


def fit_metadata(fit: SAEFit) -> dict[str, object]:
    return {
        "E": fit.E,
        "k0": fit.k0,
        "N_capacity": fit.N_capacity,
        "k": fit.k,
        "l0_target": fit.l0_target,
        "l0_actual": fit.l0_actual,
        "l0_relative_error": fit.l0_relative_error,
        "recon_R2": fit.recon_r2,
        "recon_r2": fit.recon_r2,
        "dead_frac": fit.dead_frac,
        "task_retention": fit.task_retention,
        "matched_tier": fit.matched_tier,
        "quality_warning": fit.quality_warning,
        "nearest_recon_R2": fit.nearest_recon_R2,
        "nearest_N": fit.nearest_N,
    }


def checkpoint_path_for_fit(checkpoint_dir: str | Path, fit: SAEFit, seed: int) -> Path:
    return Path(checkpoint_dir) / f"N{fit.N_capacity}_k0{fit.k0}_seed{seed}.pt"


def firing_rate_path_for_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.stem + "_firing_rate.npy")


def save_firing_rate_artifact(
    sae,
    acts_raw: torch.Tensor,
    checkpoint_path: Path,
    batch_size: int = 8192,
) -> dict[str, object]:
    """Persist per-feature firing rates next to a checkpoint.

    The 2D SAE profile needs real firing-rate vectors for cross-seed dictionary
    matching. This is intentionally computed from the same raw activations used
    by the SAE path, not from synthetic or Gaussian directions.
    """
    out_path = firing_rate_path_for_checkpoint(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(sae.parameters()).device
    fired = torch.zeros(sae.N, dtype=torch.float64, device=device)
    n_rows = 0
    with torch.no_grad():
        for start in range(0, int(acts_raw.shape[0]), batch_size):
            batch = acts_raw[start : start + batch_size].to(device)
            z = sae.encode(sae.normalise(batch))
            fired += (z > 0).sum(dim=0).to(torch.float64)
            n_rows += int(z.shape[0])
    firing_rate = (fired / max(n_rows, 1)).detach().cpu().numpy()
    tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    with tmp_path.open("wb") as f:
        np.save(f, firing_rate)
    tmp_path.replace(out_path)

    meta = {
        "firing_rate_path": str(out_path),
        "firing_rate_split": "train",
        "firing_rate_rows": int(n_rows),
        "active_feature_count": int((firing_rate > 0).sum()),
        "mean_firing_rate": float(np.mean(firing_rate)) if firing_rate.size else float("nan"),
    }
    meta_path = out_path.with_suffix(".json")
    tmp_meta = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp_meta.replace(meta_path)
    return meta


def output_missing_firing_rates(output_path: Path) -> bool:
    """Return True when a cached recon run lacks firing-rate sidecars."""
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
    if not checkpoints:
        return False
    return any(not firing_rate_path_for_checkpoint(path).exists() for path in checkpoints)


def empty_steering_metrics(reason: str) -> dict[str, object]:
    return {
        "steering_status": reason,
        "target_effect": float("nan"),
        "offtarget_damage": float("nan"),
        "selectivity": float("nan"),
        "random_selectivity": float("nan"),
        "excess_selectivity": float("nan"),
        "wbi": float("nan"),
        "random_wbi_mean": float("nan"),
        "wbi_improvement": float("nan"),
        "excess_selectivity_ci_low": float("nan"),
        "excess_selectivity_ci_high": float("nan"),
        "wbi_improvement_ci_low": float("nan"),
        "wbi_improvement_ci_high": float("nan"),
        "patient_bootstrap_valid_samples": 0,
        "patient_bootstrap_samples": 0,
        "steering_pass": False,
        "pass_fail_reason": reason,
    }


def missing_artifact_row(
    model: str,
    concept: str,
    task: str,
    layer: int,
    args: argparse.Namespace,
    error: Exception,
) -> dict[str, object]:
    return {
        "model": model,
        "concept": concept,
        "task": task,
        "layer": layer,
        "recon_target": args.recon_target,
        "E": float("nan"),
        "k0": float("nan"),
        "N_capacity": float("nan"),
        "k": float("nan"),
        "l0_target": float("nan"),
        "l0_actual": float("nan"),
        "l0_relative_error": float("nan"),
        "recon_R2": float("nan"),
        "recon_r2": float("nan"),
        "dead_frac": float("nan"),
        "task_retention": float("nan"),
        "matched_tier": "missing_artifact",
        "quality_warning": True,
        "nearest_recon_R2": float("nan"),
        "nearest_N": float("nan"),
        "steering_status": "missing_artifact",
        "error": str(error),
    }


def kappa_from_coupling(coupling: pd.DataFrame, model: str, concept: str, task: str) -> float:
    rows = coupling[
        (coupling["model"] == model)
        & (coupling["source_concept"] == concept)
        & (coupling["source_task"] == task)
    ]
    if rows.empty:
        return 0.0
    return float(rows["max_other_r2_drop"].max())


def decoded_raw_features(sae, acts_raw: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        recon_norm, _, _ = sae(acts_raw)
        return sae.denormalise(recon_norm).detach().cpu().numpy()


def run_cell(env, row: pd.Series, coupling: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    model = str(row["model"])
    concept, task, layer = parse_candidate(str(row["candidate"]))
    device = args.device
    if hasattr(env, "set_active_task"):
        env.set_active_task(task)
    stage_path = Path(args.out) / "metrics_stage.json"
    if args.skip_existing and stage_path.exists():
        with stage_path.open(encoding="utf-8") as f:
            cached_stage = json.load(f)
        checkpoint_dir = args.checkpoint_dir or str(Path(args.out) / "checkpoints")
        if "wbi" in cached_stage:
            expected_checkpoint = Path(checkpoint_dir) / (
                f"N{int(cached_stage.get('N_capacity'))}_k0{int(cached_stage.get('k0'))}_seed{args.seed}.pt"
            )
            if not expected_checkpoint.exists() or firing_rate_path_for_checkpoint(expected_checkpoint).exists():
                return cached_stage
            # Existing metrics are usable, but the profile sidecar is missing.
            # Continue so we can reload the checkpoint and write firing rates.
        elif not output_missing_firing_rates(Path(args.out) / "sae_recon_curve.csv"):
            return cached_stage

    try:
        train_acts = env.load_activations(model, layer, "train").to(device)
        test_acts = env.load_activations(model, layer, "test").to(device)
    except (KeyError, FileNotFoundError) as exc:
        result = missing_artifact_row(model, concept, task, layer, args, exc)
        if args.recon_curve_only:
            output_path = Path(args.out) / "sae_recon_curve.csv"
            tmp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
            pd.DataFrame([result]).to_csv(tmp_path, index=False)
            tmp_path.replace(output_path)
        else:
            tmp_stage = stage_path.with_name(stage_path.name + f".tmp.{os.getpid()}")
            tmp_stage.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            tmp_stage.replace(stage_path)
        return result
    selection_kwargs = {
        "device": device,
        "steps": args.steps,
        "E_grid": parse_int_grid(args.E_grid),
        "k0_grid": parse_int_grid(args.k0_grid),
        "l0_grid": parse_int_grid(args.l0_grid) if args.l0_grid else None,
        "n_features_grid": parse_int_grid(args.n_features_grid) if args.n_features_grid else None,
        "selection_mode": args.selection_mode,
        "recon_target": args.recon_target,
        "recon_band_width": args.recon_band_width,
        "relaxed_band_width": args.relaxed_band_width,
        "recon_r2_floor": args.recon_r2_floor,
        "max_dead_frac": args.max_dead_frac,
        "min_task_retention": args.min_task_retention,
        "quality_dead_frac": args.quality_dead_frac,
        "quality_retention": args.quality_retention,
        "seed": args.seed,
        "checkpoint_dir": args.checkpoint_dir or str(Path(args.out) / "checkpoints"),
        "checkpoint_every": args.checkpoint_every,
    }
    if args.recon_curve_only:
        fits = sweep_operating_points(
            train_acts,
            test_acts,
            task_retention_fn=None,
            **selection_kwargs,
        )
        rows = []
        for fit in fits:
            checkpoint_path = checkpoint_path_for_fit(selection_kwargs["checkpoint_dir"], fit, args.seed)
            firing_meta = save_firing_rate_artifact(fit.sae, train_acts, checkpoint_path)
            rows.append(
                {
                    "model": model,
                    "concept": concept,
                    "task": task,
                    "layer": layer,
                    "sae_seed": args.seed,
                    "recon_target": args.recon_target,
                    **fit_metadata(fit),
                    **firing_meta,
                    "steering_status": "recon_curve_only",
                }
            )
        curve = pd.DataFrame(rows)
        output_path = Path(args.out) / "sae_recon_curve.csv"
        tmp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
        curve.to_csv(tmp_path, index=False)
        tmp_path.replace(output_path)
        return {
            "model": model,
            "concept": concept,
            "task": task,
            "layer": layer,
            "recon_target": args.recon_target,
            "recon_curve_points": len(rows),
            "steering_status": "recon_curve_only",
            "out_csv": str(output_path),
        }

    clean_auroc = env.forward_with_patch(model, layer, "test", lambda acts: acts)

    def retention_fn(sae):
        patched = clamp_reconstruct(
            sae,
            test_acts,
            clamp_idx=[],
            centroid=np.zeros(sae.N, dtype=np.float32),
        )
        return env.forward_with_patch(model, layer, "test", lambda _acts: patched) / max(clean_auroc, 1e-9)

    fit = select_operating_point(
        train_acts,
        test_acts,
        task_retention_fn=retention_fn,
        **selection_kwargs,
    )
    if args.require_matched_tier and fit.matched_tier != args.require_matched_tier:
        raise RuntimeError(
            f"required matched_tier={args.require_matched_tier}, got {fit.matched_tier} "
            f"(nearest_recon_R2={fit.nearest_recon_R2}, nearest_N={fit.nearest_N})"
        )
    sae = fit.sae
    checkpoint_path = checkpoint_path_for_fit(selection_kwargs["checkpoint_dir"], fit, args.seed)
    firing_meta = save_firing_rate_artifact(sae, train_acts, checkpoint_path)
    W_dec = sae.decoder_directions().detach().cpu().numpy()
    U_r = env.load_leace_subspace(model, concept, task, layer)
    v_q = env.load_cav(model, concept, layer)
    with torch.no_grad():
        z_train = sae.encode(sae.normalise(train_acts)).detach().cpu().numpy()
        z_test = sae.encode(sae.normalise(test_acts)).detach().cpu().numpy()
    centroid = concept_neutral_centroid(z_train)

    measurements_train, names = env.load_measurements("train")
    measurements_test, _ = env.load_measurements("test")
    concept_col = env.concept_column(concept)
    concept_idx = names.index(concept_col)

    cav_ranking = select_target_features(v_q, W_dec, sae.N)
    concept_ranking = select_concept_features(
        z_train,
        measurements_train[:, concept_idx],
        sae.N,
    )
    if args.feature_ranking == "concept":
        ranking = concept_ranking
    elif args.feature_ranking == "cav":
        ranking = cav_ranking
    else:
        cav_decomp_probe = decomposability(
            z_train,
            z_test,
            list(cav_ranking[: args.n_features]),
            measurements_train[:, concept_idx],
            measurements_test[:, concept_idx],
        )
        concept_decomp_probe = decomposability(
            z_train,
            z_test,
            list(concept_ranking[: args.n_features]),
            measurements_train[:, concept_idx],
            measurements_test[:, concept_idx],
        )
        ranking = concept_ranking if concept_decomp_probe > cav_decomp_probe else cav_ranking
    selected = list(ranking[: args.n_features])

    cav_selected = list(cav_ranking[: args.n_features])
    concept_selected = list(concept_ranking[: args.n_features])
    A_geo_cav = geometric_agreement(U_r, W_dec, cav_selected)
    A_geo_concept_ranked = geometric_agreement(U_r, W_dec, concept_selected)
    decomp_cav = decomposability(
        z_train,
        z_test,
        cav_selected,
        measurements_train[:, concept_idx],
        measurements_test[:, concept_idx],
    )
    decomp_concept_ranked = decomposability(
        z_train,
        z_test,
        concept_selected,
        measurements_train[:, concept_idx],
        measurements_test[:, concept_idx],
    )
    decomp = decomposability(
        z_train,
        z_test,
        selected,
        measurements_train[:, concept_idx],
        measurements_test[:, concept_idx],
    )
    n90_ranking = ranking[: args.n90_max_features] if args.n90_max_features > 0 else ranking
    n90, decomp_full = sparsity_n90(
        z_train,
        z_test,
        n90_ranking,
        measurements_train[:, concept_idx],
        measurements_test[:, concept_idx],
    )
    n90_cav_ranking = cav_ranking[: args.n90_max_features] if args.n90_max_features > 0 else cav_ranking
    n90_concept_ranking = (
        concept_ranking[: args.n90_max_features] if args.n90_max_features > 0 else concept_ranking
    )
    n90_cav, decomp_full_cav = sparsity_n90(
        z_train,
        z_test,
        n90_cav_ranking,
        measurements_train[:, concept_idx],
        measurements_test[:, concept_idx],
    )
    n90_concept_ranked, decomp_full_concept_ranked = sparsity_n90(
        z_train,
        z_test,
        n90_concept_ranking,
        measurements_train[:, concept_idx],
        measurements_test[:, concept_idx],
    )

    fdec = FiducialDecoder().fit(decoded_raw_features(sae, train_acts), measurements_train, names=names)
    off_idx = [idx for idx, name in enumerate(names) if name != concept_col]

    def steer_fn(feature_idx):
        return clamp_reconstruct(sae, test_acts, clamp_idx=feature_idx, centroid=centroid)

    def target_readout(patched):
        return env.forward_with_patch(model, layer, "test", lambda _acts: patched)

    def offtarget_readout(patched):
        pred = fdec.predict(patched.detach().cpu().numpy())
        values = []
        for idx in off_idx:
            var = np.var(measurements_test[:, idx])
            if var > 1e-12:
                values.append(max(0.0, 1.0 - np.mean((measurements_test[:, idx] - pred[:, idx]) ** 2) / var))
        return float(np.mean(values)) if values else 0.0

    base_row = {
        "model": model,
        "concept": concept,
        "task": task,
        "layer": layer,
        "kappa": kappa_from_coupling(coupling, model, concept, task),
        "recon_target": args.recon_target,
        **fit_metadata(fit),
        "feature_ranking": args.feature_ranking,
        "sae_seed": args.seed,
        "A_geo": A_geo_cav,
        "A_geo_cav": A_geo_cav,
        "A_geo_concept_ranked": A_geo_concept_ranked,
        "decomposability": decomp,
        "decomposability_cav": decomp_cav,
        "decomposability_concept_ranked": decomp_concept_ranked,
        "decomposability_full_ranked": decomp_full,
        "decomposability_full_cav": decomp_full_cav,
        "decomposability_full_concept_ranked": decomp_full_concept_ranked,
        "n90": n90,
        "n90_cav": n90_cav,
        "n90_concept_ranked": n90_concept_ranked,
        "n90_feature_cap": int(len(n90_ranking)),
        **firing_meta,
    }
    if fit.matched_tier == "no_matched_point":
        result = {**base_row, **empty_steering_metrics("no_matched_point")}
        tmp_stage = stage_path.with_name(stage_path.name + f".tmp.{os.getpid()}")
        tmp_stage.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp_stage.replace(stage_path)
        return result
    tmp_stage = stage_path.with_name(stage_path.name + f".tmp.{os.getpid()}")
    tmp_stage.write_text(json.dumps(base_row, indent=2) + "\n", encoding="utf-8")
    tmp_stage.replace(stage_path)

    if args.selectivity_mode == "endpoint":
        sweep = run_endpoint_selectivity(
            steer_fn=steer_fn,
            target_readout=target_readout,
            offtarget_readout=offtarget_readout,
            ranked_features=ranking,
            n_features=sae.N,
            n_selected=args.n_features,
            n_random=args.n_random,
            seed=args.seed,
        )
    else:
        sweep = run_selectivity_sweep(
            steer_fn=steer_fn,
            target_readout=target_readout,
            offtarget_readout=offtarget_readout,
            ranked_features=ranking,
            n_features=sae.N,
            f_grid=np.linspace(0, 1, args.f_steps),
            n_random=args.n_random,
        )

    target_effect = float(sweep.get("target_effect", float("nan")))
    offtarget_damage = float(sweep.get("offtarget_damage", float("nan")))
    random_target = float(sweep.get("random_target_effect_mean", float("nan")))
    random_offtarget = float(sweep.get("random_offtarget_damage_mean", float("nan")))
    selectivity = target_effect - offtarget_damage
    random_selectivity = random_target - random_offtarget
    sweep["selectivity"] = float(selectivity)
    sweep["random_selectivity"] = float(random_selectivity)
    sweep["excess_selectivity"] = float(selectivity - random_selectivity)
    sweep["wbi_improvement"] = float(float(sweep.get("random_wbi_mean", float("nan"))) - float(sweep.get("wbi", float("nan"))))
    sweep["steering_status"] = "completed"
    sweep["pass_fail_reason"] = "bootstrap_not_run"
    sweep["steering_pass"] = False
    sweep["patient_bootstrap_samples"] = args.bootstrap_samples
    sweep["patient_bootstrap_valid_samples"] = 0
    sweep["excess_selectivity_ci_low"] = None
    sweep["excess_selectivity_ci_high"] = None
    sweep["wbi_improvement_ci_low"] = None
    sweep["wbi_improvement_ci_high"] = None
    sweep["excess_selectivity_p_one_sided"] = None
    sweep["wbi_improvement_p_one_sided"] = None
    if (
        args.bootstrap_samples > 0
        and args.selectivity_mode == "endpoint"
        and hasattr(env, "forward_scores_with_patch")
        and "selected_features" in sweep
        and "random_feature_sets" in sweep
    ):
        def score_and_pred(feature_idx: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray]:
            patched = steer_fn(feature_idx)
            scores = env.forward_scores_with_patch(model, layer, "test", lambda _acts: patched)
            pred = fdec.predict(patched.detach().cpu().numpy())
            return scores, pred

        clean_scores, clean_pred = score_and_pred([])
        concept_scores, concept_pred = score_and_pred(list(sweep["selected_features"]))
        random_score_rows = []
        random_pred = []
        for feature_set in sweep["random_feature_sets"]:
            scores_i, pred_i = score_and_pred(list(feature_set))
            random_score_rows.append(scores_i)
            random_pred.append(pred_i)
        row_idx = clean_scores["row_indices"].astype(int)
        if not np.array_equal(row_idx, concept_scores["row_indices"].astype(int)):
            raise RuntimeError("concept score rows are not aligned with clean score rows")
        for scores_i in random_score_rows:
            if not np.array_equal(row_idx, scores_i["row_indices"].astype(int)):
                raise RuntimeError("random score rows are not aligned with clean score rows")
        bootstrap = patient_bootstrap_selectivity(
            y=clean_scores["y"],
            patient_ids=clean_scores["patient_ids"],
            clean_scores=clean_scores["scores"],
            concept_scores=concept_scores["scores"],
            random_scores=[scores_i["scores"] for scores_i in random_score_rows],
            measurements=measurements_test[row_idx][:, off_idx],
            clean_pred=clean_pred[row_idx][:, off_idx],
            concept_pred=concept_pred[row_idx][:, off_idx],
            random_pred=[pred[row_idx][:, off_idx] for pred in random_pred],
            n_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        sweep.update(bootstrap)
        ci_excess = bootstrap.get("excess_selectivity_ci_low")
        ci_wbi = bootstrap.get("wbi_improvement_ci_low")
        passed = (
            ci_excess is not None
            and ci_wbi is not None
            and float(ci_excess) > 0.0
            and float(ci_wbi) > 0.0
        )
        sweep["steering_pass"] = bool(passed)
        sweep["pass_fail_reason"] = "pass" if passed else "ci_not_strictly_positive"
    elif args.bootstrap_samples > 0:
        sweep["pass_fail_reason"] = "bootstrap_unavailable"
    result = {**base_row, **sweep}
    tmp_stage = stage_path.with_name(stage_path.name + f".tmp.{os.getpid()}")
    tmp_stage.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp_stage.replace(stage_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True)
    parser.add_argument("--coupling", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--environment", choices=["stub", "benchmark", "csfm", "transformer"], default="stub")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--only-model", default="")
    parser.add_argument("--only-concept", default="")
    parser.add_argument("--only-task", default="")
    parser.add_argument("--cell-index", type=int, default=-1)
    parser.add_argument("--limit-cells", type=int, default=0)
    parser.add_argument("--max-test-shards", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--n90-max-features", type=int, default=256)
    parser.add_argument("--feature-ranking", choices=["cav", "concept", "best"], default="cav")
    parser.add_argument("--E-grid", default="4,8,16")
    parser.add_argument("--k0-grid", default="16,32,64")
    parser.add_argument("--l0-grid", default="")
    parser.add_argument("--n-features-grid", default="")
    parser.add_argument("--selection-mode", choices=["floor", "recon_band"], default="floor")
    parser.add_argument("--recon-target", type=float, default=0.90)
    parser.add_argument("--recon-band-width", type=float, default=0.02)
    parser.add_argument("--relaxed-band-width", type=float, default=0.04)
    parser.add_argument("--recon-curve-only", action="store_true")
    parser.add_argument("--require-matched-tier", default="")
    parser.add_argument("--recon-r2-floor", type=float, default=0.5)
    parser.add_argument("--max-dead-frac", type=float, default=0.30)
    parser.add_argument("--min-task-retention", type=float, default=0.98)
    parser.add_argument("--quality-dead-frac", type=float, default=0.20)
    parser.add_argument("--quality-retention", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--f-steps", type=int, default=11)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--selectivity-mode", choices=["endpoint", "curve"], default="endpoint")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=4311)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.cells)
    coupling = pd.read_csv(args.coupling)
    if args.environment == "csfm":
        from .csfm_environment import CSFMSAEEnvironment

        env = CSFMSAEEnvironment(
            artifact_root=args.artifacts,
            device=args.device,
            max_test_shards=args.max_test_shards,
        )
    elif args.environment == "transformer":
        from .transformer_environment import TransformerSAEEnvironment

        env = TransformerSAEEnvironment(
            artifact_root=args.artifacts,
            device=args.device,
            max_test_shards=args.max_test_shards,
        )
    elif args.environment == "benchmark":
        from .benchmark_environment import BenchmarkSAEEnvironment

        env = BenchmarkSAEEnvironment(artifact_root=args.artifacts)
    else:
        env = StubEnvironment(args.artifacts)

    output_path = out / "sae_layer_per_cell.csv"
    if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        if not args.recon_curve_only or not output_missing_firing_rates(out / "sae_recon_curve.csv"):
            print(f"skip existing SAE rows at {output_path}")
            return

    rows = []
    for row_idx, row in cells.iterrows():
        if args.cell_index >= 0 and int(row_idx) != args.cell_index:
            continue
        model = str(row["model"])
        concept, task, _layer = parse_candidate(str(row["candidate"]))
        if args.only_model and model != args.only_model:
            continue
        if args.only_concept and concept != args.only_concept:
            continue
        if args.only_task and task != args.only_task:
            continue
        if args.environment == "csfm" and model != "CSFM":
            continue
        if args.environment == "transformer" and model == "CSFM":
            continue
        if args.pilot and (model, concept, task) not in PILOT_CELLS:
            continue
        rows.append(run_cell(env, row, coupling, args))
        if args.limit_cells and len(rows) >= args.limit_cells:
            break

    result = pd.DataFrame(rows)
    tmp_path = out / f"sae_layer_per_cell.csv.tmp.{os.getpid()}"
    result.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    if len(result) >= 12:
        test = coupling_prior_wbi_test(result["kappa"].to_numpy(), result["wbi"].to_numpy())
    else:
        test = {"skipped": f"n={len(result)}; pilot or too small for coupling->WBI regression"}
    tmp_json = out / f"coupling_wbi_test.json.tmp.{os.getpid()}"
    tmp_json.write_text(json.dumps(test, indent=2) + "\n", encoding="utf-8")
    tmp_json.replace(out / "coupling_wbi_test.json")
    print(f"wrote {len(result)} SAE rows to {out}")


if __name__ == "__main__":
    main()
