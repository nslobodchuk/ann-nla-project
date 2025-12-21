# -*- coding: utf-8 -*-
"""
ScaNN-подобный ANN-индекс для cosine/inner product поиска.

Это учебная реализация в стиле проекта NLA (без зависимости от scann/tensorflow).
Цели:
1) Показать линейно-алгебраическую структуру ScaNN: разбиение базы + PQ/LUT + rerank.
2) Сохранить интерфейс, совместимый с ноутбуками проекта: fit(...) и search_batch(...).
3) Дать воспроизводимый код с русскими комментариями и docstring'ами.

Ограничения:
- Реализация на NumPy, ориентирована на средние размеры (десятки/сотни тысяч векторов).
- Для очень больших баз (миллионы) нужно использовать оптимизированные библиотеки.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# -----------------------------
# Вспомогательные функции
# -----------------------------
def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-нормировка по строкам."""
    X = X.astype(np.float32, copy=False)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (nrm + eps)


def _choose_split_indices(d: int, n_subspaces: int) -> np.ndarray:
    """
    Разбиение [0, d) на n_subspaces почти равных contiguous-блоков.

    В отличие от "классического" PQ, НЕ требуем d % n_subspaces == 0.
    """
    n_subspaces = int(n_subspaces)
    if n_subspaces <= 0:
        raise ValueError("n_subspaces должен быть > 0")
    # Индексы разбиения: [0 = s0 < s1 < ... < sM = d]
    splits = np.linspace(0, d, n_subspaces + 1).round().astype(int)
    # Гарантируем монотонность и границы
    splits[0] = 0
    splits[-1] = d
    for i in range(1, len(splits) - 1):
        splits[i] = max(splits[i], splits[i - 1])
    return splits


# -----------------------------
# Coarse partitioning (spherical minibatch k-means)
# -----------------------------
@dataclass
class CoarseKMeansConfig:
    """Конфигурация coarse-разбиения базы (routing)."""
    n_clusters: int = 256
    train_size: int = 50_000
    n_iters: int = 20
    batch_size: int = 4096
    random_state: int = 0


class CoarseKMeans:
    """
    Быстрый spherical k-means в minibatch-режиме для unit-векторов (cosine).
    """

    def __init__(self, cfg: CoarseKMeansConfig):
        self.cfg = cfg
        self.centroids_: Optional[np.ndarray] = None
        self.counts_: Optional[np.ndarray] = None

    def fit(self, Xn: np.ndarray) -> "CoarseKMeans":
        """
        Обучить центроиды на подвыборке базы.

        Xn: (N, d) — предполагается, что векторы L2-нормированы.
        """
        Xn = Xn.astype(np.float32, copy=False)
        n, d = Xn.shape

        rng = np.random.default_rng(self.cfg.random_state)
        train_size = int(min(self.cfg.train_size, n))
        idx = rng.choice(n, size=train_size, replace=False)
        Xtr = Xn[idx]

        k = int(self.cfg.n_clusters)
        if k <= 1:
            raise ValueError("n_clusters должен быть >= 2")
        if train_size < k:
            raise ValueError("train_size должен быть >= n_clusters")

        # Инициализация: случайные точки
        init_idx = rng.choice(train_size, size=k, replace=False)
        C = Xtr[init_idx].copy()
        C = l2_normalize(C)

        counts = np.zeros(k, dtype=np.int64)

        for _ in range(int(self.cfg.n_iters)):
            bsz = int(min(self.cfg.batch_size, train_size))
            bidx = rng.choice(train_size, size=bsz, replace=False)
            B = Xtr[bidx]  # (bsz, d)

            # Назначение по максимальному cosine (т.к. unit)
            sims = B @ C.T  # (bsz, k)
            a = np.argmax(sims, axis=1)

            # Онлайн-обновление центроидов (EMA по счетчику)
            for j in range(k):
                mask = (a == j)
                if not np.any(mask):
                    continue
                bj = B[mask]
                counts[j] += bj.shape[0]
                # Среднее по батчу + нормировка
                C[j] = C[j] + bj.mean(axis=0)
                # Нормировка на сфере
                nrm = np.linalg.norm(C[j])
                if nrm > 0:
                    C[j] /= nrm

        self.centroids_ = C.astype(np.float32, copy=False)
        self.counts_ = counts
        return self

    def assign(self, Xn: np.ndarray) -> np.ndarray:
        """Назначить каждому вектору ближайший (по cosine) coarse-центроид."""
        if self.centroids_ is None:
            raise RuntimeError("CoarseKMeans не обучен. Вызовите fit().")
        Xn = Xn.astype(np.float32, copy=False)
        sims = Xn @ self.centroids_.T
        return np.argmax(sims, axis=1).astype(np.int32)


# -----------------------------
# Anisotropic VQ / PQ
# -----------------------------
@dataclass
class AnisotropicPQConfig:
    """
    Конфигурация "анизотропного" PQ (score-aware).

    n_subspaces: число подпространств (блоков признаков).
    n_codewords: число кодвордов в каждом подпространстве.
    gamma: параметр анизотропии (gamma=1 => обычная k-means по L2).
    """
    n_subspaces: int = 10
    n_codewords: int = 64
    gamma: float = 1.1
    train_size: int = 20_000
    n_iters: int = 10
    batch_size: int = 4096
    random_state: int = 0


class AnisotropicPQ:
    """
    Anisotropic Product Quantization (APQ) для ускорения inner product/cosine через LUT.
    """

    def __init__(self, cfg: AnisotropicPQConfig):
        self.cfg = cfg
        self.splits_: Optional[np.ndarray] = None  # shape (M+1,)
        self.codebooks_: Optional[list[np.ndarray]] = None  # M x (Ks, ds_m)

    def fit(self, X: np.ndarray) -> "AnisotropicPQ":
        """
        Обучить кодбуки APQ на подвыборке базы X.

        X: (N, d), float32.
        """
        X = X.astype(np.float32, copy=False)
        n, d = X.shape

        rng = np.random.default_rng(self.cfg.random_state)
        train_size = int(min(self.cfg.train_size, n))
        idx = rng.choice(n, size=train_size, replace=False)
        Xtr = X[idx]

        M = int(self.cfg.n_subspaces)
        Ks = int(self.cfg.n_codewords)
        if Ks <= 1:
            raise ValueError("n_codewords должен быть >= 2")

        splits = _choose_split_indices(d, M)
        codebooks: list[np.ndarray] = []

        for m in range(M):
            s0, s1 = int(splits[m]), int(splits[m + 1])
            Xm = Xtr[:, s0:s1]
            C = _anisotropic_kmeans(
                Xm,
                k=Ks,
                gamma=float(self.cfg.gamma),
                n_iters=int(self.cfg.n_iters),
                batch_size=int(self.cfg.batch_size),
                rng=rng,
            )
            codebooks.append(C.astype(np.float32, copy=False))

        self.splits_ = splits
        self.codebooks_ = codebooks
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Закодировать векторы X в PQ-коды.

        Возвращает:
            codes: (N, M) int32, где M=n_subspaces.
        """
        if self.splits_ is None or self.codebooks_ is None:
            raise RuntimeError("AnisotropicPQ не обучен. Вызовите fit().")

        X = X.astype(np.float32, copy=False)
        n, d = X.shape
        M = len(self.codebooks_)
        codes = np.empty((n, M), dtype=np.int32)

        for m in range(M):
            s0, s1 = int(self.splits_[m]), int(self.splits_[m + 1])
            Xm = X[:, s0:s1]
            C = self.codebooks_[m]
            codes[:, m] = _assign_anisotropic(Xm, C, gamma=float(self.cfg.gamma))

        return codes

    def lut(self, Q: np.ndarray) -> list[np.ndarray]:
        """
        Построить LUT (таблицы скалярных произведений) для пачки запросов Q.

        Возвращает список длины M:
            tables[m]: (nq, Ks), где tables[m][i, k] = <q_i^{(m)}, c_k^{(m)}>
        """
        if self.splits_ is None or self.codebooks_ is None:
            raise RuntimeError("AnisotropicPQ не обучен. Вызовите fit().")
        Q = Q.astype(np.float32, copy=False)
        nq, d = Q.shape

        tables: list[np.ndarray] = []
        for m, C in enumerate(self.codebooks_):
            s0, s1 = int(self.splits_[m]), int(self.splits_[m + 1])
            Qm = Q[:, s0:s1]
            tables.append(Qm @ C.T)  # (nq, Ks)
        return tables

    def approx_scores_from_codes(self, lut_tables: list[np.ndarray], codes: np.ndarray) -> np.ndarray:
        """
        Быстро посчитать приближённые скоры для набора кодов (ADC с LUT).

        lut_tables: список M таблиц (nq, Ks)
        codes: (Ncand, M)
        Возвращает: (nq, Ncand) float32
        """
        nq = lut_tables[0].shape[0]
        Ncand, M = codes.shape
        scores = np.zeros((nq, Ncand), dtype=np.float32)
        for m in range(M):
            tbl = lut_tables[m]  # (nq, Ks)
            cm = codes[:, m]     # (Ncand,)
            scores += tbl[:, cm]
        return scores


def _assign_anisotropic(X: np.ndarray, C: np.ndarray, gamma: float, eps: float = 1e-12) -> np.ndarray:
    """
    Назначение точек X к центроидам C по анизотропному loss.
    Векторизовано по точкам; по центроидам — матрично.

    loss(x,c) = ||x-c||^2 + (gamma-1) * ((||x||^2 - <x,c>)^2 / (||x||^2 + eps))
    """
    X = X.astype(np.float32, copy=False)
    C = C.astype(np.float32, copy=False)
    x2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    c2 = np.sum(C * C, axis=1, keepdims=True).T  # (1,k)
    dot = X @ C.T  # (n,k)
    dist2 = x2 + c2 - 2.0 * dot

    if gamma != 1.0:
        penalty = (x2 - dot) ** 2 / (x2 + eps)
        dist2 = dist2 + (gamma - 1.0) * penalty

    return np.argmin(dist2, axis=1).astype(np.int32)


def _anisotropic_kmeans(
    X: np.ndarray,
    k: int,
    gamma: float,
    n_iters: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    K-means по анизотропному функционалу.

    Обновление центроида для кластера S:
        (|S| I + (gamma-1) * Σ (x x^T / (||x||^2 + eps))) c = gamma * Σ x

    Это явная "вычислительная линейная алгебра": на каждой итерации решаются
    малые СЛАУ размерности ds x ds (ds — размерность подпространства).
    """
    X = X.astype(np.float32, copy=False)
    n, ds = X.shape
    if n < k:
        raise ValueError("k больше числа точек")

    # Инициализация: случайные точки
    init_idx = rng.choice(n, size=k, replace=False)
    C = X[init_idx].copy()

    eps = 1e-12
    for _ in range(int(n_iters)):
        # Для больших n можно использовать minibatch; здесь делаем простую батчевую схему.
        # Assign (полный проход)
        a = _assign_anisotropic(X, C, gamma=gamma, eps=eps)

        # Update
        for j in range(k):
            idx = np.where(a == j)[0]
            if idx.size == 0:
                # Реинициализация пустого кластера
                C[j] = X[rng.integers(0, n)]
                continue

            S = X[idx]  # (nj, ds)
            nj = S.shape[0]
            # Нормы по подпространству
            s2 = np.sum(S * S, axis=1) + eps  # (nj,)

            # A = nj*I + (gamma-1) * Σ (x x^T / ||x||^2)
            A = np.eye(ds, dtype=np.float32) * float(nj)
            if gamma != 1.0:
                # Σ (x x^T / s2) = (S^T * (1/s2)) @ S
                W = (1.0 / s2).astype(np.float32)  # (nj,)
                A = A + (gamma - 1.0) * (S.T * W) @ S  # (ds, ds)

            # b = gamma * Σ x
            b = float(gamma) * np.sum(S, axis=0).astype(np.float32)  # (ds,)

            # Решаем СЛАУ
            try:
                C[j] = np.linalg.solve(A, b).astype(np.float32, copy=False)
            except np.linalg.LinAlgError:
                # На случай вырождения — псевдорешение
                C[j] = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32, copy=False)

    return C.astype(np.float32, copy=False)


# -----------------------------
# Полный ScaNN-like индекс
# -----------------------------
@dataclass
class ScannConfig:
    """Конфигурация ScaNN-подобного индекса."""
    # coarse routing
    nlist: int = 256
    nprobe: int = 8

    # PQ
    n_subspaces: int = 10
    n_codewords: int = 64
    gamma: float = 1.1

    # обучение (подвыборки)
    coarse_train_size: int = 50_000
    coarse_n_iters: int = 20
    pq_train_size: int = 20_000
    pq_n_iters: int = 10

    # rerank
    candidate_count: int = 2000  # сколько кандидатов брать после coarse routing (до PQ)
    reorder_k: int = 200         # сколько лучших по PQ отправлять на точный rerank

    random_state: int = 0


class ScannIndex:
    """
    ScaNN-подобный индекс:
    1) Coarse разбиение базы на nlist кластеров (routing).
    2) APQ кодирование базы + LUT-скоринг.
    3) Rerank точным cosine для shortlist.

    Предполагается cosine similarity на unit-векторах (т.е. inner product после нормировки).
    """

    def __init__(self, cfg: ScannConfig):
        self.cfg = cfg

        self.Xn_: Optional[np.ndarray] = None
        self.coarse_: Optional[CoarseKMeans] = None
        self.assign_: Optional[np.ndarray] = None
        self.invlists_: Optional[list[np.ndarray]] = None

        self.pq_: Optional[AnisotropicPQ] = None
        self.codes_: Optional[np.ndarray] = None

    def fit(self, Xn: np.ndarray, assume_normalized: bool = True) -> "ScannIndex":
        """
        Построить индекс по базе.

        Xn: (N, d). Если assume_normalized=False, то будет выполнена l2_normalize.
        """
        Xn = Xn.astype(np.float32, copy=False)
        if not assume_normalized:
            Xn = l2_normalize(Xn)

        self.Xn_ = Xn

        # Coarse
        c_cfg = CoarseKMeansConfig(
            n_clusters=int(self.cfg.nlist),
            train_size=int(self.cfg.coarse_train_size),
            n_iters=int(self.cfg.coarse_n_iters),
            random_state=int(self.cfg.random_state),
        )
        coarse = CoarseKMeans(c_cfg).fit(Xn)
        assign = coarse.assign(Xn)  # (N,)

        invlists = []
        for j in range(int(self.cfg.nlist)):
            invlists.append(np.where(assign == j)[0].astype(np.int32))

        # PQ
        pq_cfg = AnisotropicPQConfig(
            n_subspaces=int(self.cfg.n_subspaces),
            n_codewords=int(self.cfg.n_codewords),
            gamma=float(self.cfg.gamma),
            train_size=int(self.cfg.pq_train_size),
            n_iters=int(self.cfg.pq_n_iters),
            random_state=int(self.cfg.random_state),
        )
        pq = AnisotropicPQ(pq_cfg).fit(Xn)
        codes = pq.encode(Xn)  # (N, M)

        self.coarse_ = coarse
        self.assign_ = assign
        self.invlists_ = invlists
        self.pq_ = pq
        self.codes_ = codes
        return self

    def search_batch(
        self,
        Qn: np.ndarray,
        top_k: int,
        exclude_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск top_k ближайших для батча запросов.

        Qn: (nq, d) unit-вектора.
        exclude_ids: (nq,) индексы в базе, которые нужно исключить (например self-match).
        Возвращает:
            pred_ids: (nq, top_k)
            pred_scores: (nq, top_k) — точные cosine на этапе rerank.
        """
        if self.Xn_ is None or self.coarse_ is None or self.invlists_ is None or self.pq_ is None or self.codes_ is None:
            raise RuntimeError("Индекс не обучен. Вызовите fit().")

        Qn = Qn.astype(np.float32, copy=False)
        nq, d = Qn.shape
        top_k = int(top_k)

        # LUT для PQ
        tables = self.pq_.lut(Qn)

        # Coarse routing: для каждого q выбираем top-nprobe центроидов
        C = self.coarse_.centroids_  # (nlist, d)
        sims_coarse = Qn @ C.T  # (nq, nlist)
        probes = np.argpartition(-sims_coarse, kth=min(self.cfg.nprobe, sims_coarse.shape[1]-1)-1, axis=1)[:, : int(self.cfg.nprobe)]

        pred_ids = np.empty((nq, top_k), dtype=np.int32)
        pred_scores = np.empty((nq, top_k), dtype=np.float32)

        for i in range(nq):
            pr = probes[i]
            # Собираем кандидатов из инвертированных списков
            cand = np.concatenate([self.invlists_[int(j)] for j in pr], axis=0)
            if cand.size == 0:
                pred_ids[i] = -1
                pred_scores[i] = -np.inf
                continue

            # Ограничим размер кандидатов (для скорости)
            if cand.size > int(self.cfg.candidate_count):
                # Берём кандидатов из более похожих кластеров приоритетно
                # (простая эвристика: сортировка probes по coarse similarity)
                pr_sorted = pr[np.argsort(-sims_coarse[i, pr])]
                collected = []
                for j in pr_sorted:
                    collected.append(self.invlists_[int(j)])
                    if sum(len(x) for x in collected) >= int(self.cfg.candidate_count):
                        break
                cand = np.concatenate(collected, axis=0)[: int(self.cfg.candidate_count)]

            # PQ скоринг (ADC)
            cand_codes = self.codes_[cand]  # (Nc, M)
            # scores_pq: (1, Nc)
            scores_pq = self.pq_.approx_scores_from_codes([t[i:i+1] for t in tables], cand_codes)[0]

            # shortlist по PQ
            rk = int(min(self.cfg.reorder_k, cand.size))
            top_idx = np.argpartition(-scores_pq, kth=rk-1)[:rk]
            shortlist = cand[top_idx]

            # точный rerank
            exact = (Qn[i] @ self.Xn_[shortlist].T).astype(np.float32, copy=False)

            # self-exclude
            if exclude_ids is not None:
                ex = int(exclude_ids[i])
                if ex >= 0:
                    mask = (shortlist == ex)
                    if np.any(mask):
                        exact[mask] = -np.inf

            # top_k
            kk = min(top_k, shortlist.size)
            best = np.argpartition(-exact, kth=kk-1)[:kk]
            # сортировка по убыванию
            order = best[np.argsort(-exact[best])]

            ids = shortlist[order].astype(np.int32, copy=False)
            sc = exact[order].astype(np.float32, copy=False)

            # если кандидатов < top_k, дополним -1/-inf
            if kk < top_k:
                pad = top_k - kk
                ids = np.concatenate([ids, -np.ones(pad, dtype=np.int32)])
                sc = np.concatenate([sc, -np.inf * np.ones(pad, dtype=np.float32)])

            pred_ids[i] = ids[:top_k]
            pred_scores[i] = sc[:top_k]

        return pred_ids, pred_scores
