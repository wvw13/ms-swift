"""Functions preserved from the validated legacy evaluator. See PROVENANCE.json."""
import os,json
import numpy as np
from . import data as ds
from . import metrics as mt

def l2n_np(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)

def aggregate(per_image: np.ndarray, modality: str) -> np.ndarray:
    """C = CFP; O = mean of 6 OCT vectors then L2N; B = 0.5*C+0.5*O then L2N.

    Input per-image vectors are individually L2-normalized.
    """
    cfp = per_image[:, 0, :]
    octs = per_image[:, 1:, :]
    if modality == "C":
        return cfp
    if modality == "O":
        return l2n_np(octs.mean(axis=1))
    if modality == "B":
        o = l2n_np(octs.mean(axis=1))
        return l2n_np(0.5 * cfp + 0.5 * o)
    raise ValueError(modality)

def multi_hot(cases: list) -> np.ndarray:
    m = np.zeros((len(cases), ds.N_CLASSES), dtype=np.float32)
    for i, c in enumerate(cases):
        m[i, c.labels] = 1.0
    return m

def run_text_retrieval(q_vec: np.ndarray, t_vec: np.ndarray, q_cases: list, ks, tag: str, run_dir: str):
    """Experiment 1: case (C/O/B) -> 22 disease texts."""
    q_vec = l2n_np(q_vec.astype(np.float64))
    t_vec = l2n_np(t_vec.astype(np.float64))
    sims = q_vec @ t_vec.T
    gains_all = multi_hot(q_cases)          # [N,22] 0/1 gain per candidate
    tiebreak = np.arange(ds.N_CLASSES)      # class id ascending
    per_query, rows = [], []
    for i, c in enumerate(q_cases):
        order = mt.rank_candidates(sims[i], np.ones(ds.N_CLASSES, bool), tiebreak)
        res = mt.evaluate_query(gains_all[i, order], gains_all[i], ks)
        res["sample_path"] = c.sample_path
        res["labels"] = c.labels
        per_query.append(res)
        top5 = [
            {"class_id": int(j), "label": ds.CLASSES[int(j)]["label"], "score": float(sims[i, j]),
             "gain": float(gains_all[i, j])}
            for j in order[:5]
        ]
        rows.append({"sample_path": c.sample_path, "labels": c.labels, "top5": top5, "metrics": res})
    summary = mt.aggregate_query_results(per_query)
    summary.update({"task": tag, "gallery": "22 disease names", "n_gallery": ds.N_CLASSES})
    _write_task_outputs(run_dir, tag, rows, sims, [c.sample_path for c in q_cases],
                        [c["label"] for c in ds.CLASSES])
    return summary

def _write_task_outputs(run_dir, tag, rows, sims, q_ids, g_ids):
    tag_clean = tag.replace("->", "_").replace(">", "_").replace("/", "_")
    np.savez_compressed(os.path.join(run_dir, f"scores__{tag_clean}.npz"),
                        sims=sims.astype(np.float32), query_ids=q_ids, gallery_ids=g_ids)
    with open(os.path.join(run_dir, f"per_query__{tag_clean}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    return tag_clean

def exclusion_masks(q_cases: list, g_cases: list, cfg: dict) -> tuple:
    """Per-query valid-gallery masks: exclude self / same patient / duplicate images.

    Duplicate = the two cases share at least one image file with identical
    content (md5), checked across all 7 images of each case.
    """
    ds.attach_md5s(q_cases + g_cases, cfg)
    masks = np.ones((len(q_cases), len(g_cases)), dtype=bool)
    reasons = {"self": 0, "same_patient": 0, "duplicate_image": 0}
    g_index = {c.sample_path: j for j, c in enumerate(g_cases)}
    g_patients = [f"{c.center}|{c.patient_key}" for c in g_cases]
    g_md5 = [set(c.md5s) for c in g_cases]
    for i, qc in enumerate(q_cases):
        qi = g_index.get(qc.sample_path)
        if qi is not None:
            masks[i, qi] = False
            reasons["self"] += 1
        qkey = f"{qc.center}|{qc.patient_key}"
        for j, gp in enumerate(g_patients):
            if masks[i, j] and gp == qkey:
                masks[i, j] = False
                reasons["same_patient"] += 1
        for j, gm in enumerate(g_md5):
            if masks[i, j] and not gm.isdisjoint(qc.md5s):
                masks[i, j] = False
                reasons["duplicate_image"] += 1
    return masks, reasons

def run_case_retrieval(q_vec, g_vec, q_cases, g_cases, ks, tag, run_dir, cfg, same_model=True):
    """Experiment 2 / 3 case-level retrieval with exclusions and Jaccard gains."""
    assert same_model
    q_vec = l2n_np(q_vec.astype(np.float64))
    g_vec = l2n_np(g_vec.astype(np.float64))
    sims = q_vec @ g_vec.T
    masks, reasons = exclusion_masks(q_cases, g_cases, cfg)
    # gains: Jaccard between label sets (0/1 for text task is not used here)
    mq, mg = multi_hot(q_cases), multi_hot(g_cases)
    inter = mq @ mg.T
    union = mq.sum(1, keepdims=True) + mg.sum(1, keepdims=True).T - inter
    gains = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    # deterministic tiebreak: lexicographic rank of normalized path
    g_sorted_rank = np.argsort(np.argsort(np.array([c.sample_path for c in g_cases])))
    per_query, rows = [], []
    for i, qc in enumerate(q_cases):
        order = mt.rank_candidates(sims[i], masks[i], g_sorted_rank)
        valid_gains = gains[i][masks[i]]
        res = mt.evaluate_query(gains[i, order], valid_gains, ks)
        res["sample_path"] = qc.sample_path
        res["labels"] = qc.labels
        res["n_excluded"] = int((~masks[i]).sum())
        per_query.append(res)
        top5 = [
            {"sample_path": g_cases[int(j)].sample_path, "score": float(sims[i, j]),
             "jaccard_gain": float(gains[i, j]), "labels": g_cases[int(j)].labels}
            for j in order[:5]
        ]
        rows.append({"sample_path": qc.sample_path, "labels": qc.labels, "top5": top5, "metrics": res})
    summary = mt.aggregate_query_results(per_query)
    summary.update({
        "task": tag,
        "n_gallery": len(g_cases),
        "exclusions_total": {k: int(v) for k, v in reasons.items()},
        "n_candidates_excluded_per_query_mean": float(np.mean((~masks).sum(1))) if len(q_cases) else 0.0,
    })
    _write_task_outputs(run_dir, tag, rows, sims,
                        [c.sample_path for c in q_cases], [c.sample_path for c in g_cases])
    return summary
