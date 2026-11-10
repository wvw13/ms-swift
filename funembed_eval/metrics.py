"""Retrieval metrics: Hit@K, Recall@K, NDCG@K with K=1,3,5.

Protocol (fixed by the evaluation plan):
- Similarity: cosine on L2-normalized vectors, descending order.
- Tie rule: disease candidates by class id (ascending); case candidates by
  normalized sample path lexicographic order (ascending).
- Relevance gains:
  * image->text: 1.0 if the candidate disease is in the query's label set else 0.
  * case retrieval Hit/Recall: binary "related" = the two cases share at least
    one label (two no_abnormality cases therefore relate, normal vs abnormal do not).
  * case retrieval NDCG: Jaccard similarity of the two label sets as linear gain.
- Recall denominator: number of relevant candidates in the WHOLE valid gallery
  (for text retrieval this equals the query's number of true labels).
- DCG@K = sum(gain / log2(rank+1)), rank starting at 1, linear gain, no 2^g-1.
- IDCG@K computed from the whole valid gallery sorted by true gain, not from
  the model's returned top-K.
- Queries with zero relevant candidates are excluded from the three metric
  means and reported separately.
"""

from __future__ import annotations

import numpy as np


def rank_candidates(scores: np.ndarray, valid_mask: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Rank valid candidates for one query.

    Args:
        scores: [G] similarity scores (higher = better).
        valid_mask: [G] bool, False = candidate excluded before ranking.
        tiebreak: [G] int key used for deterministic tie handling; ties in
            score are broken by ascending tiebreak key (class id or the rank
            of the normalized sample path in lexicographic order).

    Returns:
        Array of candidate indices in ranked (best first) order.
    """
    idx = np.where(valid_mask)[0]
    order = np.lexsort((tiebreak[idx], -scores[idx].astype(np.float64)))
    return idx[order]


def dcg_at_k(ranked_gains: np.ndarray, k: int) -> float:
    """DCG@K with linear gain: sum(gain_i / log2(i+1)), i = 1..K."""
    k = min(k, len(ranked_gains))
    if k <= 0:
        return 0.0
    ranks = np.arange(1, k + 1, dtype=np.float64)
    return float(np.sum(np.asarray(ranked_gains, dtype=np.float64)[:k] / np.log2(ranks + 1.0)))


def idcg_at_k(all_gains: np.ndarray, k: int) -> float:
    """IDCG@K from the whole valid gallery sorted by true gain (descending)."""
    top = np.sort(np.asarray(all_gains, dtype=np.float64))[::-1][:k]
    ranks = np.arange(1, len(top) + 1, dtype=np.float64)
    return float(np.sum(top / np.log2(ranks + 1.0)))


def evaluate_query(ranked_gains: np.ndarray, all_gains: np.ndarray, ks=(1, 3, 5)) -> dict:
    """Metrics for one query.

    Args:
        ranked_gains: relevance gains of the ranked (model order) valid candidates.
        all_gains: gains over ALL valid candidates (unranked); used for the
            recall denominator and IDCG.
    """
    ranked_gains = np.asarray(ranked_gains, dtype=np.float64)
    all_gains = np.asarray(all_gains, dtype=np.float64)
    n_relevant = int(np.sum(all_gains > 0))
    if n_relevant == 0:
        return {"n_relevant": 0, "no_relevant": True}

    out = {"n_relevant": n_relevant, "no_relevant": False}
    for k in ks:
        top = ranked_gains[:k]
        out[f"hit@{k}"] = float(np.any(top > 0))
        out[f"recall@{k}"] = float(np.sum(top > 0) / n_relevant)
        idcg = idcg_at_k(all_gains, k)
        out[f"ndcg@{k}"] = dcg_at_k(ranked_gains, k) / idcg if idcg > 0 else 0.0
    return out


def aggregate_query_results(results: list) -> dict:
    """Equal-weight mean over queries that have at least one relevant candidate.

    Queries flagged ``no_relevant`` are counted and reported separately.
    """
    valid = [r for r in results if not r.get("no_relevant", False)]
    n_no_rel = len(results) - len(valid)
    summary = {
        "n_queries_total": len(results),
        "n_queries_valid": len(valid),
        "n_queries_no_relevant": n_no_rel,
    }
    if not valid:
        return summary
    keys = [k for k in valid[0] if k.startswith(("hit@", "recall@", "ndcg@"))]
    for k in keys:
        summary[k] = float(np.mean([r[k] for r in valid]))
    return summary


def jaccard_gain(labels_a: set, labels_b: set) -> float:
    """Jaccard similarity |A∩B| / |A∪B| between two label-id sets.

    Two no_abnormality cases get 1.0; normal vs abnormal gets 0.0.
    """
    union = labels_a | labels_b
    if not union:
        return 0.0
    return len(labels_a & labels_b) / len(union)


def binary_related(labels_a: set, labels_b: set) -> bool:
    """Related for Hit/Recall: share at least one label (incl. no_abnormality)."""
    return len(labels_a & labels_b) > 0
