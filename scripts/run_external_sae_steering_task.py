#!/usr/bin/env python
"""Run frozen-atom and local-atom SAE steering on one external native task."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCALE = ROOT / "results/sae_reconciliation/matched_scale_v1"
BASE = ROOT / "results/external_benchmark_v1"
ACT_ROOT = ROOT / "results/activations_external_full_v1/pooled"
TRANSPORT = ROOT / "results/multicohort/pooled_sae_transport/pooled_transport_model_cohort_gate.csv"
FAMILIES = {
    "af_rhythm_native": "rate_rhythm", "bbb_conduction_native": "conduction",
    "qt_interval_native": "interval", "st_t_abnormal_native": "st_t",
    "af_rhythm_icd": "rate_rhythm", "bbb_conduction_icd": "conduction",
    "qt_interval_icd": "interval", "mi_ischemia_icd": "mi_ischemia",
    "hypertrophy_icd": "hypertrophy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-suffix", required=True); p.add_argument("--cohort", required=True)
    p.add_argument("--target", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--sae-source", choices=("source", "cohort_adapted"), default="source")
    return p.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def threshold_at_specificity(y: np.ndarray, score: np.ndarray, specificity: float = 0.95) -> float:
    neg = score[y == 0]
    return float(np.quantile(neg, specificity, method="higher"))


def encode(sae, x: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    import torch
    chunks = []
    sae.eval()
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            raw = torch.as_tensor(np.asarray(x[lo:lo+batch]), dtype=torch.float32, device=device)
            chunks.append(sae.encode(raw).cpu().numpy())
    return np.concatenate(chunks)


def load_activations(index_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    from scripts.train_external_frozen_heads import load_activations as loader
    return loader(index_dir)


def metrics(y: np.ndarray, clean: np.ndarray, edited: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {
        "baseline_auroc": float(roc_auc_score(y, clean)), "edited_auroc": float(roc_auc_score(y, edited)),
        "auroc_drop": float(roc_auc_score(y, clean) - roc_auc_score(y, edited)),
        "baseline_auprc": float(average_precision_score(y, sigmoid(clean))),
        "edited_auprc": float(average_precision_score(y, sigmoid(edited))),
        "baseline_sens_at_95spec": float((clean[y == 1] >= threshold).mean()),
        "edited_sens_at_95spec": float((edited[y == 1] >= threshold).mean()),
    }


def main() -> None:
    a = parse_args()
    import torch
    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    cohort = a.cohort.lower().replace("-", "_")
    head_root = BASE / a.model_suffix / cohort
    bundle = joblib.load(head_root / "frozen_heads.joblib")
    if a.target not in bundle["heads"]:
        metrics_path = head_root / "frozen_heads_metrics.csv"
        reason = "head_not_trained"
        detail = {}
        if metrics_path.exists():
            head_metrics = pd.read_csv(metrics_path)
            match = head_metrics[head_metrics.task.eq(a.target)]
            if not match.empty:
                detail = match.iloc[0].dropna().to_dict()
                reason = str(detail.get("status", reason))
        skipped = head_root / "steering" / "skipped" / f"seed{a.seed}" / f"{a.target}.json"
        skipped.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "model_suffix": a.model_suffix,
            "cohort": cohort,
            "target": a.target,
            "seed": a.seed,
            "status": "skipped",
            "reason": reason,
            "head_metrics": detail,
        }
        tmp = skipped.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(skipped)
        print(json.dumps(payload))
        return
    names = list(bundle["targets"]); heads = bundle["heads"]; scaler = bundle["scaler"]
    labels = {name: np.asarray(heads[name]["labels"], dtype=float) for name in names}
    split = np.asarray(bundle["split"]); record_ids = np.asarray(bundle["record_ids"])
    group_ids = np.asarray(bundle.get("group_ids", record_ids))
    x, loaded_ids = load_activations(ACT_ROOT / a.model_suffix / cohort)
    if not np.array_equal(record_ids.astype(str), loaded_ids.astype(str)):
        raise RuntimeError("Activation/head record order mismatch")
    tr, va, te = np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]

    training = pd.read_csv(SCALE / "training_manifest.csv")
    source_row = training[(training.feature_suffix == a.model_suffix) & (training.seed == a.seed)].iloc[0]
    model = source_row.model; safe = model.lower().replace("-", "_")
    if a.sae_source == "source":
        checkpoint = Path(source_row.checkpoint)
    else:
        adapted = pd.read_csv(BASE / "cohort_adapted_sae_manifest.csv")
        adapted_row = adapted[
            (adapted.model_suffix == a.model_suffix) & (adapted.cohort == cohort) & (adapted.seed == a.seed)
        ].iloc[0]
        checkpoint = Path(adapted_row.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cohort-adapted SAE checkpoint missing: {checkpoint}")
    saved = torch.load(checkpoint, map_location=a.device); cfg = saved["config"]
    n_features, sae_k = int(cfg["n_features"]), int(cfg["k"])
    cache = head_root / "steering_cache" / a.sae_source / f"seed{a.seed}_N{n_features}_k{sae_k}"
    complete = cache / "complete.json"; lock = cache.with_name(cache.name + ".lock")
    expected = {
        "model": model, "cohort": cohort, "seed": a.seed, "sae_source": a.sae_source,
        "checkpoint": str(checkpoint), "targets": names,
    }

    def valid_cache() -> bool:
        try: return complete.exists() and json.loads(complete.read_text()) == expected
        except (OSError, json.JSONDecodeError): return False

    builder = False
    if not valid_cache():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try: os.mkdir(lock); builder = True
        except FileExistsError:
            for _ in range(1200):
                if valid_cache(): break
                time.sleep(1)
            else: raise RuntimeError(f"timeout waiting for {cache}")
    if builder:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            sae = BatchTopKSAE(x.shape[1], n_features, sae_k).to(a.device)
            sae.load_state_dict(saved["model"]); sae.eval()
            ztr, zva, zte = encode(sae, x[tr], a.device), encode(sae, x[va], a.device), encode(sae, x[te], a.device)
            decoder = sae.W_dec.detach().cpu().numpy(); sigma = sae.sigma.detach().cpu().numpy()
            gradients = []; rankings = []
            for name in names:
                coef = np.asarray(heads[name]["clf"].coef_).reshape(-1)
                gradient = ((coef / scaler.scale_) * sigma) @ decoder
                gradients.append(gradient)
                focus = labels[name][tr] == 1
                rankings.append(np.argsort((ztr[focus] * gradient).mean(0))[::-1])
            gradients = np.asarray(gradients); rankings = np.asarray(rankings)
            with torch.no_grad():
                recon_va = sae.decode(torch.as_tensor(zva, device=a.device)).cpu().numpy()
                recon_te = sae.decode(torch.as_tensor(zte, device=a.device)).cpu().numpy()
            base_va = np.column_stack([heads[n]["clf"].decision_function(scaler.transform(recon_va)) for n in names])
            base = np.column_stack([heads[n]["clf"].decision_function(scaler.transform(recon_te)) for n in names])
            thresholds = np.asarray([threshold_at_specificity(labels[n][va].astype(int), base_va[:,j]) for j,n in enumerate(names)])
            arrays = {
                "zte": zte.astype(np.float32), "gradients": gradients.astype(np.float64),
                "rankings": rankings.astype(np.int32), "freq": (ztr > 0).mean(0).astype(np.float32),
                "mag": np.divide(ztr.sum(0), (ztr > 0).sum(0), out=np.zeros(n_features), where=(ztr > 0).sum(0)>0).astype(np.float32),
                "centroid": ztr.mean(0).astype(np.float32), "base": base.astype(np.float32),
                "thresholds": thresholds.astype(np.float32),
            }
            for key,value in arrays.items():
                tmp = cache / f"{key}.npy.tmp.{os.getpid()}"
                with tmp.open("wb") as handle: np.save(handle,value)
                tmp.replace(cache / f"{key}.npy")
            tmp = cache / f"complete.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(expected,sort_keys=True)+"\n"); tmp.replace(complete)
        finally:
            try: os.rmdir(lock)
            except OSError: pass

    zte=np.load(cache/"zte.npy",mmap_mode="r"); gradients=np.load(cache/"gradients.npy",mmap_mode="r")
    rankings=np.load(cache/"rankings.npy",mmap_mode="r"); freq=np.load(cache/"freq.npy",mmap_mode="r")
    mag=np.load(cache/"mag.npy",mmap_mode="r"); centroid=np.load(cache/"centroid.npy",mmap_mode="r")
    base=np.load(cache/"base.npy",mmap_mode="r"); thresholds=np.load(cache/"thresholds.npy",mmap_mode="r")
    local_rank = rankings[names.index(a.target)]
    if a.sae_source == "source":
        frozen = json.loads((BASE/"frozen_atom_registry"/safe/f"seed{a.seed}.json").read_text())["tasks"][a.target]
        protocol_atoms = {
            "frozen_atom": {k:np.asarray(frozen[k],dtype=int) for k in ("top1","top5","top10")},
            "local_atom": {f"top{k}":local_rank[:k].astype(int) for k in (1,5,10)},
        }
    else:
        protocol_atoms = {
            "cohort_adapted_atom": {f"top{k}":local_rank[:k].astype(int) for k in (1,5,10)},
        }
    excluded = np.unique(np.concatenate([rankings[j,:10] for j in range(len(names))]))

    def delta(indices: np.ndarray) -> np.ndarray:
        idx=np.asarray(indices,dtype=int); dz=centroid[idx][None,:]-zte[:,idx]
        return np.column_stack([dz @ gradients[j,idx] for j in range(len(names))])

    transport = pd.read_csv(TRANSPORT)
    gate = transport[(transport.model==model)&(transport.cohort==cohort)].iloc[0]
    source_profile = pd.read_csv(SCALE/"matched_scale_model_profile.csv").set_index("model").loc[model]
    if a.sae_source == "source":
        dictionary_fidelity_eligible = bool(source_profile.matched_scale_primary_eligible)
        transport_eligible = bool(gate.primary_transport_eligible)
        transport_gate_applicable = True
        dictionary_metrics = {}
    else:
        metrics_path = checkpoint.with_suffix(".metrics.json")
        dictionary_metrics = json.loads(metrics_path.read_text())
        dictionary_fidelity_eligible = bool(float(dictionary_metrics["explained_variance"]) >= 0.90)
        transport_eligible = True
        transport_gate_applicable = False
    for protocol,selected in protocol_atoms.items():
        out=head_root/"steering"/protocol/f"seed{a.seed}"/a.target; out.mkdir(parents=True,exist_ok=True)
        final=out/"result.json"; records=out/"records.npz"
        if final.exists() and records.exists(): continue
        rng=np.random.default_rng(a.seed+sum(map(ord,a.target+protocol)))
        random_groups=[]
        for _ in range(a.n_random):
            group=[]
            for atom in selected["top5"]:
                dist=np.abs(np.log((freq+1e-6)/(freq[atom]+1e-6)))+np.abs(np.log((mag+1e-6)/(mag[atom]+1e-6)))
                blocked=np.unique(np.concatenate([excluded,np.asarray(group,dtype=int)])); dist[blocked]=np.inf
                group.append(int(rng.choice(np.argsort(dist)[:200])))
            random_groups.append(group)
        random_groups=np.asarray(random_groups,dtype=int)
        deltas={key:delta(value) for key,value in selected.items()}; random_delta=np.stack([delta(g) for g in random_groups],axis=1)
        metric={}
        for key,d in deltas.items():
            metric[key]={name:metrics(labels[name][te].astype(int),base[:,j],base[:,j]+d[:,j],float(thresholds[j])) for j,name in enumerate(names)}
        raw_metric=float(heads[a.target]["metrics"]["test_auroc"]); recon_metric=float(metric["top5"][a.target]["baseline_auroc"])
        payload={
            "schema_version":1,"model":model,"model_suffix":a.model_suffix,"cohort":cohort,"target":a.target,
            "target_type":"binary","family":FAMILIES[a.target],"seed":a.seed,"protocol":protocol,
            "checkpoint":str(checkpoint),"selected_atoms":{k:v.tolist() for k,v in selected.items()},
            "dictionary_source":a.sae_source,
            "dictionary_training_recon_r2":dictionary_metrics.get("explained_variance"),
            "dictionary_training_dead_fraction":dictionary_metrics.get("dead_fraction"),
            "dictionary_quality_warning":bool(
                a.sae_source == "cohort_adapted"
                and float(dictionary_metrics.get("dead_fraction", 0.0)) > 0.20
            ),
            "random_groups":random_groups.tolist(),"focus_thresholds_train":{n:1.0 for n in names},"metrics":metric,
            "raw_head_test_auroc":raw_metric,"sae_recon_head_test_auroc":recon_metric,
            "sae_readout_retention":recon_metric/max(raw_metric,1e-8),
            "source_fidelity_eligible":dictionary_fidelity_eligible,
            "transport_eligible":transport_eligible,
            "transport_gate_applicable":transport_gate_applicable,
            "split_unit":bundle.get("split_unit","record"),
            "claim_tier":"primary_patient_level" if bundle.get("split_unit")=="patient" else "secondary_record_level_sensitivity",
        }
        tmp=final.with_suffix(f".json.tmp.{os.getpid()}"); tmp.write_text(json.dumps(payload,indent=2)+"\n"); tmp.replace(final)
        label_matrix=np.column_stack([labels[n][te] for n in names]).astype(np.float32)
        tmp=out/f"records.npz.tmp.{os.getpid()}"
        with tmp.open("wb") as handle:
            np.savez(handle,patient_ids=group_ids[te].astype("U64"),target_names=np.asarray(names,dtype="U64"),
                target_types=np.asarray(["binary"]*len(names),dtype="U16"),labels=label_matrix,
                baseline_logits=np.asarray(base,dtype=np.float32),top1_delta=deltas["top1"].astype(np.float32),
                top5_delta=deltas["top5"].astype(np.float32),top10_delta=deltas["top10"].astype(np.float32),
                random_top5_delta=random_delta.astype(np.float32),thresholds_95spec=np.asarray(thresholds,dtype=np.float32),
                continuous_target_means=np.full(len(names),np.nan),continuous_target_stds=np.full(len(names),np.nan))
        tmp.replace(records)
        print(json.dumps({"model":model,"cohort":cohort,"target":a.target,"seed":a.seed,"protocol":protocol,"top5":selected["top5"].tolist()}))


if __name__ == "__main__":
    main()
