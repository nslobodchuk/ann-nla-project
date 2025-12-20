from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hashlib
import numpy as np


def compute_dataset_id(
    n_base: int,
    seed: int,
    q_idx_all: np.ndarray,
    q_idx_test: np.ndarray,
    version: str = "v1",
) -> str:
    """
    Stable dataset identifier to prevent silent mixing of artifacts.
    """
    h = hashlib.sha256()
    h.update(version.encode("utf-8"))
    h.update(str(int(n_base)).encode("utf-8"))
    h.update(str(int(seed)).encode("utf-8"))
    h.update(np.ascontiguousarray(q_idx_all, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(q_idx_test, dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _topk_desc_1d(a: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= a.size:
        return np.argsort(-a)
    idx = np.argpartition(-a, kth=k - 1)[:k]
    return idx[np.argsort(-a[idx])]


def _right_singular_vectors_rand(
    Y: np.ndarray,
    r: int,
    rng: np.random.Generator,
    oversample: int = 8,
    n_iter: int = 2,
) -> np.ndarray:
    """
    Approximate top-right singular vectors of Y using randomized SVD.
    Returns V_r as (m, r), where Y is (n, m).

    Notes:
    - Includes QR re-orthonormalization during power iterations for stability.
    """
    n, m = Y.shape
    r = int(min(r, m))
    if r <= 0:
        return np.empty((m, 0), dtype=np.float32)

    k = int(min(m, r + max(0, oversample)))
    Omega = rng.standard_normal(size=(m, k), dtype=np.float32)  # (m, k)
    Z = Y @ Omega                                               # (n, k)

    # Power iterations with re-orthonormalization
    for _ in range(int(n_iter)):
        Z, _ = np.linalg.qr(Z, mode="reduced")
        Z = Y @ (Y.T @ Z)

    Q, _ = np.linalg.qr(Z, mode="reduced")                      # (n, k)
    B = Q.T @ Y                                                 # (k, m)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)            # Vt: (k, m)
    return Vt[:r].T.astype(np.float32, copy=False)              # (m, r)


@dataclass
class _ClusterModel:
    ids: np.ndarray                 # (m_l,), int32 indices into corpus
    A: Optional[np.ndarray]         # (d, r_l), float32
    B: Optional[np.ndarray]         # (r_l, m_l), float32
    used_fallback: bool             # True if exact fallback scoring is used


class LorannIndex:
    """
    LoRANN-style search for cosine/IP:
      - IVF routing via k-means on unit vectors
      - per-cluster Reduced-Rank Regression scoring: y_hat = (q^T A_l) B_l
      - shortlist by approximate scores, then exact rerank on corpus vectors
    """

    def __init__(
        self,
        n_clusters: int = 1024,
        rank: int = 32,
        n_probe: int = 16,
        candidate_count: int = 1000,
        max_train_per_cluster: int = 2048,
        min_train_per_cluster: int = 64,
        seed: int = 42,
        randomized_svd: bool = True,
        svd_oversample: int = 8,
        svd_n_iter: int = 2,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.rank = int(rank)
        self.n_probe = int(n_probe)
        self.candidate_count = int(candidate_count)
        self.max_train_per_cluster = int(max_train_per_cluster)
        self.min_train_per_cluster = int(min_train_per_cluster)

        self.randomized_svd = bool(randomized_svd)
        self.svd_oversample = int(svd_oversample)
        self.svd_n_iter = int(svd_n_iter)

        self._rng = np.random.default_rng(int(seed))

        self.X: Optional[np.ndarray] = None             # (N, d), float32, normalized
        self.centroids: Optional[np.ndarray] = None     # (L, d), float32, normalized
        self.models: Optional[List[_ClusterModel]] = None

        # Filled after fit()
        self.fit_stats_: Dict[str, Any] = {}

    # -------------------- public API --------------------

    def fit(
        self,
        X: np.ndarray,
        train_X: Optional[np.ndarray] = None,
        kmeans_max_iter: int = 50,
        kmeans_batch_size: int = 8192,
    ) -> "LorannIndex":
        """
        Builds IVF clusters and fits per-cluster RRR models.
        X is the corpus. train_X defaults to X (corpus-as-training).
        """
        Xn = l2_normalize(X)
        N, d = Xn.shape
        if not (2 <= self.n_clusters <= N):
            raise ValueError("n_clusters must be in [2, N].")

        self.X = Xn

        centroids, labels = self._kmeans_minibatch(
            Xn,
            n_clusters=self.n_clusters,
            max_iter=int(kmeans_max_iter),
            batch_size=int(kmeans_batch_size),
        )
        self.centroids = l2_normalize(centroids)

        cluster_ids = [np.nonzero(labels == l)[0].astype(np.int32) for l in range(self.n_clusters)]

        train_Xn = Xn if train_X is None else l2_normalize(train_X)
        train_rows = self._assign_training_rows(train_Xn, n_probe_train=self.n_probe)

        models: List[_ClusterModel] = []
        fallback_count = 0
        train_sizes: List[int] = []
        cluster_sizes: List[int] = []

        for l in range(self.n_clusters):
            ids = cluster_ids[l]
            m_l = int(ids.size)
            cluster_sizes.append(m_l)

            if m_l == 0:
                models.append(_ClusterModel(ids=ids, A=None, B=None, used_fallback=True))
                fallback_count += 1
                train_sizes.append(0)
                continue

            J = train_rows[l]
            train_sizes.append(len(J))

            # Fallback if not enough training data or degenerate cluster
            if len(J) < self.min_train_per_cluster or self.rank <= 0 or m_l < 2:
                models.append(_ClusterModel(ids=ids, A=None, B=None, used_fallback=True))
                fallback_count += 1
                continue

            if len(J) > self.max_train_per_cluster:
                J = self._rng.choice(J, size=self.max_train_per_cluster, replace=False).tolist()

            C = Xn[ids]                                        # (m_l, d)
            Xt = train_Xn[np.asarray(J, dtype=np.int64)]        # (n_l, d)
            Y = Xt @ C.T                                       # (n_l, m_l)

            r_l = int(min(self.rank, m_l))
            Vr = self._right_singular_vectors(Y, r_l)          # (m_l, r_l)
            A = (C.T @ Vr).astype(np.float32, copy=False)      # (d, r_l)
            B = (Vr.T).astype(np.float32, copy=False)          # (r_l, m_l)

            models.append(_ClusterModel(ids=ids, A=A, B=B, used_fallback=False))

        self.models = models

        self.fit_stats_ = {
            "n_clusters": int(self.n_clusters),
            "fallback_clusters": int(fallback_count),
            "fallback_frac": float(fallback_count) / float(self.n_clusters),
            "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "avg_train_per_cluster": float(np.mean(train_sizes)) if train_sizes else 0.0,
            "min_train_per_cluster_observed": int(min(train_sizes)) if train_sizes else 0,
            "max_train_per_cluster_observed": int(max(train_sizes)) if train_sizes else 0,
        }
        return self

    def search(
        self,
        q: np.ndarray,
        top_k: int = 10,
        candidate_count: Optional[int] = None,
        exclude_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches one query. Returns (ids, scores) where scores are cosine similarities.
        """
        qn = l2_normalize(q.reshape(1, -1))[0]
        return self._search_normalized(qn, top_k=top_k, candidate_count=candidate_count, exclude_id=exclude_id)

    def search_batch(
        self,
        Q: np.ndarray,
        top_k: int = 10,
        candidate_count: Optional[int] = None,
        exclude_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search. Returns (ids, scores) with shapes (nq, top_k).
        """
        Qn = l2_normalize(Q)
        nq = int(Qn.shape[0])
        top_k = int(top_k)

        if exclude_ids is None:
            exclude_ids = [None] * nq
        if len(exclude_ids) != nq:
            raise ValueError("exclude_ids must be None or have length nq.")

        ids_out = np.full((nq, top_k), -1, dtype=np.int32)
        sc_out = np.full((nq, top_k), -np.inf, dtype=np.float32)

        for i in range(nq):
            ids, sc = self._search_normalized(
                Qn[i],
                top_k=top_k,
                candidate_count=candidate_count,
                exclude_id=exclude_ids[i],
            )
            if ids.size:
                ids_out[i, : ids.size] = ids
                sc_out[i, : sc.size] = sc

        return ids_out, sc_out

    # -------------------- internal helpers --------------------

    def _right_singular_vectors(self, Y: np.ndarray, r: int) -> np.ndarray:
        if self.randomized_svd:
            return _right_singular_vectors_rand(
                Y.astype(np.float32, copy=False),
                r=r,
                rng=self._rng,
                oversample=self.svd_oversample,
                n_iter=self.svd_n_iter,
            )
        _, _, Vt = np.linalg.svd(Y, full_matrices=False)
        return Vt[: int(r)].T.astype(np.float32, copy=False)

    def _route(self, qn: np.ndarray, n_probe: int) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Index is not fitted.")
        sims = qn @ self.centroids.T
        p = int(min(int(n_probe), sims.size))
        return _topk_desc_1d(sims, p).astype(np.int32, copy=False)

    def _search_normalized(
        self,
        qn: np.ndarray,
        top_k: int,
        candidate_count: Optional[int],
        exclude_id: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.centroids is None or self.models is None:
            raise RuntimeError("Index is not fitted.")

        if candidate_count is None:
            candidate_count = self.candidate_count

        probe = self._route(qn, self.n_probe)

        ids_all: List[np.ndarray] = []
        sc_all: List[np.ndarray] = []

        for l in probe:
            mdl = self.models[int(l)]
            ids = mdl.ids
            if ids.size == 0:
                continue

            if mdl.A is not None and mdl.B is not None:
                sc = (qn @ mdl.A) @ mdl.B
            else:
                sc = qn @ self.X[ids].T  # exact fallback

            ids_all.append(ids)
            sc_all.append(sc.astype(np.float32, copy=False))

        if not ids_all:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

        ids_cat = np.concatenate(ids_all, axis=0).astype(np.int32, copy=False)
        sc_cat = np.concatenate(sc_all, axis=0).astype(np.float32, copy=False)

        if exclude_id is not None:
            m = ids_cat != int(exclude_id)
            ids_cat = ids_cat[m]
            sc_cat = sc_cat[m]

        if ids_cat.size == 0:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

        cc = int(min(int(candidate_count), ids_cat.size))
        cand_pos = _topk_desc_1d(sc_cat, cc)
        cand_ids = ids_cat[cand_pos]

        exact = qn @ self.X[cand_ids].T
        k = int(min(int(top_k), cand_ids.size))
        top_pos = _topk_desc_1d(exact, k)

        return cand_ids[top_pos], exact[top_pos].astype(np.float32, copy=False)

    def _assign_training_rows(self, train_Xn: np.ndarray, n_probe_train: int) -> List[List[int]]:
        """
        Assign each training vector to the best-matching centroid among the top n_probe_train.
        Returns list of training row indices per cluster.
        """
        if self.centroids is None:
            raise RuntimeError("Index is not fitted.")
        L = int(self.n_clusters)
        out: List[List[int]] = [[] for _ in range(L)]
        for i in range(train_Xn.shape[0]):
            q = train_Xn[i]
            probe = self._route(q, n_probe=int(n_probe_train))
            # Choose the best of probed clusters
            best = int(probe[0]) if probe.size else 0
            out[best].append(int(i))
        return out

    def _kmeans_minibatch(
        self,
        Xn: np.ndarray,
        n_clusters: int,
        max_iter: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple minibatch k-means on unit vectors.
        Returns (centroids, labels).
        """
        N, d = Xn.shape
        n_clusters = int(n_clusters)
        max_iter = int(max_iter)
        batch_size = int(batch_size)

        # init centroids from random points
        init_ids = self._rng.choice(N, size=n_clusters, replace=False)
        C = Xn[init_ids].copy()  # (L, d)
        C = l2_normalize(C)

        counts = np.zeros((n_clusters,), dtype=np.int64)

        for _ in range(max_iter):
            bidx = self._rng.integers(0, N, size=min(batch_size, N), dtype=np.int64)
            B = Xn[bidx]                          # (b, d)
            sims = B @ C.T                        # (b, L)
            assign = np.argmax(sims, axis=1)

            for i, l in enumerate(assign):
                l = int(l)
                counts[l] += 1
                eta = 1.0 / float(counts[l])
                C[l] = (1.0 - eta) * C[l] + eta * B[i]

            C = l2_normalize(C)

        labels = np.argmax(Xn @ C.T, axis=1).astype(np.int32)
        return C.astype(np.float32, copy=False), labels

