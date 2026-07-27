"""Metrics for the causally anchored SAE extension."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score


def geometric_agreement(U_r: np.ndarray, W_dec: np.ndarray, feature_idx: list[int]) -> float:
    U = np.asarray(U_r, dtype=float)
    W = np.asarray(W_dec, dtype=float)[:, list(feature_idx)]
    if U.ndim != 2 or W.ndim != 2:
        raise ValueError("U_r and W_dec[:, feature_idx] must be matrices")
    rank = np.linalg.matrix_rank(W)
    if rank == 0 or U.shape[1] == 0:
        return 0.0
    Q, _ = np.linalg.qr(W)
    Q = Q[:, :rank]
    return float(((Q.T @ U) ** 2).sum() / U.shape[1])


def select_target_features(v_q: np.ndarray, W_dec: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(v_q, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12)
    W = np.asarray(W_dec, dtype=float)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return np.argsort(-(np.abs(v @ W)))[:n]


def select_concept_features(z_train: np.ndarray, y_train: np.ndarray, n: int) -> np.ndarray:
    """Rank SAE features by train-split univariate association with a concept.

    This is intentionally separate from CAV/LEACE geometry. It answers whether
    the learned sparse code contains features that directly decode the clinical
    measurement, even when those features are not the closest decoder directions
    to the dense CAV.
    """
    z = np.asarray(z_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    zc = z - z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    z_norm = np.sqrt(np.sum(zc**2, axis=0))
    y_norm = float(np.sqrt(np.sum(yc**2)))
    denom = z_norm * max(y_norm, 1e-12)
    corr = np.divide(zc.T @ yc, denom, out=np.zeros(z.shape[1], dtype=float), where=denom > 1e-12)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(-np.abs(corr))[:n]


def concept_neutral_centroid(z_train: np.ndarray) -> np.ndarray:
    return np.asarray(z_train).mean(axis=0)


def clamp_reconstruct(sae, acts_raw: torch.Tensor, clamp_idx: list[int], centroid: np.ndarray) -> torch.Tensor:
    sae.eval()
    with torch.no_grad():
        z = sae.encode(sae.normalise(acts_raw))
        if clamp_idx:
            values = torch.as_tensor(centroid[clamp_idx], dtype=z.dtype, device=z.device)
            z[:, clamp_idx] = values.unsqueeze(0)
        recon_norm = sae.decode(z)
        return sae.denormalise(recon_norm)


class FiducialDecoder:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.names: list[str] | None = None

    def fit(self, features: np.ndarray, measurements: np.ndarray, names: list[str] | None = None):
        self.model.fit(features, measurements)
        self.names = names
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def readout_r2(self, features: np.ndarray, measurements: np.ndarray) -> dict[str | int, float]:
        pred = self.predict(features)
        out: dict[str | int, float] = {}
        for idx in range(measurements.shape[1]):
            key = self.names[idx] if self.names is not None else idx
            out[key] = float(r2_score(measurements[:, idx], pred[:, idx]))
        return out


def decomposability(
    z_train: np.ndarray,
    z_test: np.ndarray,
    feature_idx: list[int],
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> float:
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    mu = float(np.nanmean(y_train))
    sigma = float(np.nanstd(y_train))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return float("nan")
    y_train_z = np.nan_to_num((y_train - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0)
    y_test_z = np.nan_to_num((y_test - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0)
    model = Ridge(alpha=alpha).fit(z_train[:, feature_idx], y_train_z)
    pred = model.predict(z_test[:, feature_idx])
    return float(r2_score(y_test_z, pred))


def sparsity_n90(
    z_train: np.ndarray,
    z_test: np.ndarray,
    ranked_features: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> tuple[int, float]:
    ranked = list(ranked_features)
    full = decomposability(z_train, z_test, ranked, y_train, y_test, alpha=alpha)
    target = 0.9 * full
    for n in range(1, len(ranked) + 1):
        score = decomposability(z_train, z_test, ranked[:n], y_train, y_test, alpha=alpha)
        if score >= target:
            return n, full
    return len(ranked), full


def damage_curve(readout_curve: list[float]) -> np.ndarray:
    values = np.asarray(readout_curve, dtype=float)
    return values[0] - values


def selectivity_from_damage(
    target_curve: list[float],
    offtarget_curve: list[float],
    f_grid: np.ndarray,
) -> float:
    target_damage = damage_curve(target_curve)
    offtarget_damage = damage_curve(offtarget_curve)
    return float(np.trapz(offtarget_damage - target_damage, f_grid))


def wbi(target_effect: float, offtarget_damage: float, eps: float = 1e-6) -> float:
    return float(offtarget_damage / (target_effect + eps))


def _mean_offtarget_r2(measurements: np.ndarray, pred: np.ndarray) -> float:
    values = []
    for idx in range(measurements.shape[1]):
        var = float(np.var(measurements[:, idx]))
        if var > 1e-12:
            score = 1.0 - float(np.mean((measurements[:, idx] - pred[:, idx]) ** 2)) / var
            values.append(max(0.0, score))
    return float(np.mean(values)) if values else 0.0


def patient_bootstrap_selectivity(
    y: np.ndarray,
    patient_ids: np.ndarray,
    clean_scores: np.ndarray,
    concept_scores: np.ndarray,
    random_scores: list[np.ndarray],
    measurements: np.ndarray,
    clean_pred: np.ndarray,
    concept_pred: np.ndarray,
    random_pred: list[np.ndarray],
    n_samples: int = 1000,
    seed: int = 4311,
) -> dict[str, object]:
    """Patient-level paired bootstrap for concept-vs-random SAE selectivity."""
    if n_samples <= 0:
        return {
            "patient_bootstrap_samples": 0,
            "patient_bootstrap_valid_samples": 0,
            "excess_selectivity_ci_low": None,
            "excess_selectivity_ci_high": None,
            "wbi_improvement_ci_low": None,
            "wbi_improvement_ci_high": None,
            "excess_selectivity_p_one_sided": None,
            "wbi_improvement_p_one_sided": None,
        }
    y = np.asarray(y, dtype=np.int32)
    patient_ids = np.asarray(patient_ids)
    clean_scores = np.asarray(clean_scores, dtype=np.float64)
    concept_scores = np.asarray(concept_scores, dtype=np.float64)
    random_scores = [np.asarray(scores, dtype=np.float64) for scores in random_scores]
    measurements = np.asarray(measurements, dtype=np.float64)
    clean_pred = np.asarray(clean_pred, dtype=np.float64)
    concept_pred = np.asarray(concept_pred, dtype=np.float64)
    random_pred = [np.asarray(pred, dtype=np.float64) for pred in random_pred]
    unique_patients = np.asarray(sorted(set(patient_ids.tolist())), dtype=object)
    patient_groups = [np.where(patient_ids == patient)[0] for patient in unique_patients]
    rng = np.random.default_rng(seed)
    excess_values = []
    wbi_improvement_values = []

    def one_eval(idx: np.ndarray) -> tuple[float, float] | None:
        yb = y[idx]
        if len(set(yb.tolist())) < 2:
            return None
        clean_auc = float(roc_auc_score(yb, clean_scores[idx]))
        concept_auc = float(roc_auc_score(yb, concept_scores[idx]))
        clean_off = _mean_offtarget_r2(measurements[idx], clean_pred[idx])
        concept_off = _mean_offtarget_r2(measurements[idx], concept_pred[idx])
        concept_target_effect = clean_auc - concept_auc
        concept_off_damage = clean_off - concept_off
        concept_selectivity = concept_target_effect - concept_off_damage
        concept_wbi = wbi(concept_target_effect, concept_off_damage)
        random_selectivities = []
        random_wbis = []
        for scores, pred in zip(random_scores, random_pred):
            random_auc = float(roc_auc_score(yb, scores[idx]))
            random_off = _mean_offtarget_r2(measurements[idx], pred[idx])
            random_target_effect = clean_auc - random_auc
            random_off_damage = clean_off - random_off
            random_selectivities.append(random_target_effect - random_off_damage)
            random_wbis.append(wbi(random_target_effect, random_off_damage))
        if not random_selectivities:
            return None
        return (
            float(concept_selectivity - np.mean(random_selectivities)),
            float(np.mean(random_wbis) - concept_wbi),
        )

    for _ in range(n_samples):
        sampled_groups = rng.integers(0, len(patient_groups), size=len(patient_groups))
        idx = np.concatenate([patient_groups[int(group_i)] for group_i in sampled_groups])
        values = one_eval(idx)
        if values is None:
            continue
        excess, wbi_improvement = values
        if np.isfinite(excess) and np.isfinite(wbi_improvement):
            excess_values.append(excess)
            wbi_improvement_values.append(wbi_improvement)
    if not excess_values:
        return {
            "patient_bootstrap_samples": n_samples,
            "patient_bootstrap_valid_samples": 0,
            "excess_selectivity_ci_low": None,
            "excess_selectivity_ci_high": None,
            "wbi_improvement_ci_low": None,
            "wbi_improvement_ci_high": None,
            "excess_selectivity_p_one_sided": None,
            "wbi_improvement_p_one_sided": None,
        }
    excess_arr = np.asarray(excess_values, dtype=np.float64)
    wbi_arr = np.asarray(wbi_improvement_values, dtype=np.float64)
    return {
        "patient_bootstrap_samples": n_samples,
        "patient_bootstrap_valid_samples": int(len(excess_arr)),
        "excess_selectivity_ci_low": float(np.percentile(excess_arr, 2.5)),
        "excess_selectivity_ci_high": float(np.percentile(excess_arr, 97.5)),
        "wbi_improvement_ci_low": float(np.percentile(wbi_arr, 2.5)),
        "wbi_improvement_ci_high": float(np.percentile(wbi_arr, 97.5)),
        "excess_selectivity_p_one_sided": float((np.sum(excess_arr <= 0.0) + 1.0) / (len(excess_arr) + 1.0)),
        "wbi_improvement_p_one_sided": float((np.sum(wbi_arr <= 0.0) + 1.0) / (len(wbi_arr) + 1.0)),
    }


def run_selectivity_sweep(
    steer_fn,
    target_readout,
    offtarget_readout,
    ranked_features: np.ndarray,
    n_features: int,
    f_grid: np.ndarray | None = None,
    n_random: int = 20,
    seed: int = 4311,
) -> dict[str, object]:
    if f_grid is None:
        f_grid = np.linspace(0, 1, 11)
    rng = np.random.default_rng(seed)

    def curves(order: np.ndarray) -> tuple[list[float], list[float]]:
        target_values = []
        offtarget_values = []
        for frac in f_grid:
            n = int(np.floor(frac * n_features))
            patched = steer_fn(list(order[:n]))
            target_values.append(float(target_readout(patched)))
            offtarget_values.append(float(offtarget_readout(patched)))
        return target_values, offtarget_values

    target_curve, offtarget_curve = curves(ranked_features)
    delta_tcav = selectivity_from_damage(target_curve, offtarget_curve, f_grid)
    random_deltas = []
    random_target_effects = []
    random_offtarget_damages = []
    random_wbis = []
    for _ in range(n_random):
        rt, ro = curves(rng.permutation(n_features))
        random_deltas.append(selectivity_from_damage(rt, ro, f_grid))
        rt_effect = damage_curve(rt)[-1]
        ro_effect = damage_curve(ro)[-1]
        random_target_effects.append(float(rt_effect))
        random_offtarget_damages.append(float(ro_effect))
        random_wbis.append(wbi(float(rt_effect), float(ro_effect)))

    target_effect = damage_curve(target_curve)[-1]
    offtarget_effect = damage_curve(offtarget_curve)[-1]
    return {
        "target_curve": target_curve,
        "offtarget_curve": offtarget_curve,
        "delta_tcav": delta_tcav,
        "delta_tilde": float(delta_tcav - np.mean(random_deltas)),
        "target_effect": float(target_effect),
        "offtarget_damage": float(offtarget_effect),
        "wbi": wbi(float(target_effect), float(offtarget_effect)),
        "random_target_effects": random_target_effects,
        "random_offtarget_damages": random_offtarget_damages,
        "random_wbis": random_wbis,
        "random_target_effect_mean": float(np.mean(random_target_effects)) if random_target_effects else float("nan"),
        "random_offtarget_damage_mean": (
            float(np.mean(random_offtarget_damages)) if random_offtarget_damages else float("nan")
        ),
        "random_wbi_mean": float(np.mean(random_wbis)) if random_wbis else float("nan"),
        "random_wbi_sd_within_seed": float(np.std(random_wbis, ddof=1)) if len(random_wbis) > 1 else float("nan"),
    }


def run_endpoint_selectivity(
    steer_fn,
    target_readout,
    offtarget_readout,
    ranked_features: np.ndarray,
    n_features: int,
    n_selected: int,
    n_random: int = 20,
    seed: int = 4311,
) -> dict[str, object]:
    """Endpoint-only selectivity with same-size random feature-set controls."""
    rng = np.random.default_rng(seed)
    n_selected_requested = int(n_selected)
    n_selected_effective = min(n_selected_requested, int(n_features))
    selected = np.asarray(ranked_features[:n_selected_effective], dtype=int)
    unpatched = steer_fn([])
    target_base = float(target_readout(unpatched))
    offtarget_base = float(offtarget_readout(unpatched))

    patched = steer_fn(list(selected))
    target_final = float(target_readout(patched))
    offtarget_final = float(offtarget_readout(patched))
    target_effect = target_base - target_final
    offtarget_damage = offtarget_base - offtarget_final

    random_target_effects = []
    random_offtarget_damages = []
    random_wbis = []
    random_feature_sets = []
    population = np.arange(n_features)
    for _ in range(n_random):
        random_idx = rng.choice(population, size=n_selected_effective, replace=False)
        random_feature_sets.append(random_idx.astype(int).tolist())
        random_patched = steer_fn(list(random_idx))
        rt_effect = target_base - float(target_readout(random_patched))
        ro_effect = offtarget_base - float(offtarget_readout(random_patched))
        random_target_effects.append(float(rt_effect))
        random_offtarget_damages.append(float(ro_effect))
        random_wbis.append(wbi(float(rt_effect), float(ro_effect)))

    return {
        "target_curve": [target_base, target_final],
        "offtarget_curve": [offtarget_base, offtarget_final],
        "delta_tcav": float("nan"),
        "delta_tilde": float("nan"),
        "target_effect": float(target_effect),
        "offtarget_damage": float(offtarget_damage),
        "wbi": wbi(float(target_effect), float(offtarget_damage)),
        "random_target_effects": random_target_effects,
        "random_offtarget_damages": random_offtarget_damages,
        "random_wbis": random_wbis,
        "selected_features": selected.astype(int).tolist(),
        "n_selected_requested": n_selected_requested,
        "n_selected_effective": n_selected_effective,
        "requested_group_available": bool(n_selected_requested <= int(n_features)),
        "random_feature_sets": random_feature_sets,
        "random_target_effect_mean": float(np.mean(random_target_effects)) if random_target_effects else float("nan"),
        "random_offtarget_damage_mean": (
            float(np.mean(random_offtarget_damages)) if random_offtarget_damages else float("nan")
        ),
        "random_wbi_mean": float(np.mean(random_wbis)) if random_wbis else float("nan"),
        "random_wbi_sd_within_seed": float(np.std(random_wbis, ddof=1)) if len(random_wbis) > 1 else float("nan"),
    }


def coupling_prior_wbi_test(kappa: np.ndarray, wbi_values: np.ndarray) -> dict[str, object]:
    kappa = np.asarray(kappa, dtype=float)
    wbi_values = np.asarray(wbi_values, dtype=float)
    finite = np.isfinite(kappa) & np.isfinite(wbi_values)
    kappa = kappa[finite]
    wbi_values = wbi_values[finite]
    if len(kappa) < 12:
        return {"warning": f"n={len(kappa)} too small for slope; report qualitatively only"}
    design = np.vstack([np.ones_like(kappa), kappa]).T
    coef, *_ = np.linalg.lstsq(design, wbi_values, rcond=None)
    b0, b1 = coef
    resid = wbi_values - design @ coef
    denom = np.sum((kappa - kappa.mean()) ** 2)
    se = np.sqrt(np.sum(resid**2) / max(len(wbi_values) - 2, 1) / max(denom, 1e-12))
    rho, p = spearmanr(kappa, wbi_values)
    k_hi = kappa >= np.median(kappa)
    w_hi = wbi_values >= np.median(wbi_values)
    return {
        "beta0": float(b0),
        "beta1": float(b1),
        "beta1_se": float(se),
        "beta1_ci95": [float(b1 - 1.96 * se), float(b1 + 1.96 * se)],
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "off_diagonal": {
            "high_kappa_low_wbi": int(np.sum(k_hi & ~w_hi)),
            "low_kappa_high_wbi": int(np.sum(~k_hi & w_hi)),
        },
    }
