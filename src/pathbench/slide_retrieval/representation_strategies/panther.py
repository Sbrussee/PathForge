from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PantherAssignment:
    """PANTHER assignment outputs for one slide-level bag."""

    resp: np.ndarray
    top1: np.ndarray
    top1_prob: np.ndarray
    slide_embed: np.ndarray
    proto_mean: np.ndarray
    proto_cov: np.ndarray
    proto_prob: np.ndarray


class PantherPrototypeAssigner:
    """
    Apply fixed prototypes to one patch-feature bag using PANTHER EM updates.

    This mirrors the PBMS refactor path:
    fixed prototypes -> PANTHER allcat slide representation -> mixture
    probability/mean/covariance -> per-patch responsibilities.
    """

    def __init__(
        self,
        *,
        prototypes: np.ndarray,
        em_iter: int = 10,
        tau: float = 10.0,
        ot_eps: float = 0.1,
    ) -> None:
        prototype_matrix = np.asarray(prototypes, dtype=np.float32)
        if prototype_matrix.ndim != 2:
            raise ValueError(
                f"Expected prototypes with shape (P, D), got {prototype_matrix.shape}."
            )
        if prototype_matrix.shape[0] == 0:
            raise ValueError("At least one prototype is required.")
        if prototype_matrix.shape[1] == 0:
            raise ValueError("Prototype feature dimension must be > 0.")

        self.prototypes = prototype_matrix
        self.em_iter = int(em_iter)
        self.tau = float(tau)
        self.ot_eps = float(ot_eps)

        if self.em_iter < 1:
            raise ValueError("em_iter must be >= 1.")
        if self.tau < 0.0:
            raise ValueError("tau must be >= 0.")
        if self.ot_eps <= 0.0:
            raise ValueError("ot_eps must be > 0.")

    def assign(self, features: np.ndarray) -> PantherAssignment:
        z = np.asarray(features, dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"Expected features with shape (N, D), got {z.shape}.")
        if z.shape[0] == 0:
            raise ValueError("Cannot run PANTHER assignment on an empty feature bag.")
        if int(z.shape[1]) != int(self.prototypes.shape[1]):
            raise ValueError(
                "Feature and prototype dimensions must match. "
                f"Got features dim {z.shape[1]} and prototype dim {self.prototypes.shape[1]}."
            )

        prob, mean, cov = self._fit_mixture(z)
        resp = _compute_resp_diag_gauss(z, prob, mean, cov)
        top1 = np.argmax(resp, axis=1).astype(np.int32, copy=False)
        top1_prob = resp[np.arange(resp.shape[0]), top1].astype(np.float32, copy=False)
        slide_embed = np.concatenate(
            [
                prob.reshape(-1),
                mean.reshape(-1),
                cov.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        return PantherAssignment(
            resp=resp.astype(np.float32, copy=False),
            top1=top1,
            top1_prob=top1_prob,
            slide_embed=slide_embed,
            proto_mean=mean.astype(np.float32, copy=False),
            proto_cov=cov.astype(np.float32, copy=False),
            proto_prob=prob.astype(np.float32, copy=False),
        )

    def _fit_mixture(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prototypes = self.prototypes.astype(np.float64, copy=False)
        data = z.astype(np.float64, copy=False)
        n_proto, feature_dim = prototypes.shape

        prob = np.full(n_proto, 1.0 / n_proto, dtype=np.float64)
        mean = prototypes.copy()
        prior_cov = np.full((n_proto, feature_dim), self.ot_eps, dtype=np.float64)
        cov = prior_cov.copy()
        data_squared = data * data
        prototype_second_moment = prior_cov + prototypes * prototypes

        for _ in range(self.em_iter):
            resp = _compute_resp_diag_gauss(data, prob, mean, cov).astype(
                np.float64,
                copy=False,
            )
            weight_sum = resp.sum(axis=0)
            weight_sum_reg = weight_sum + self.tau

            weighted_sum = resp.T @ data
            weighted_square_sum = resp.T @ data_squared

            prob = weight_sum_reg / np.sum(weight_sum_reg)
            mean = (
                weighted_sum + prototypes * self.tau
            ) / weight_sum_reg[:, None]
            second_moment = (
                weighted_square_sum + prototype_second_moment * self.tau
            ) / weight_sum_reg[:, None]
            cov = np.clip(second_moment - mean * mean, 1e-8, None)

        return (
            prob.astype(np.float32, copy=False),
            mean.astype(np.float32, copy=False),
            cov.astype(np.float32, copy=False),
        )


def _compute_resp_diag_gauss(
    features: np.ndarray,
    prob: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    z = np.asarray(features, dtype=np.float64)
    pi = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, None)
    mu = np.asarray(mean, dtype=np.float64)
    sigma = np.clip(np.asarray(cov, dtype=np.float64), 1e-8, None)

    inv_cov = 1.0 / sigma
    log_pi = np.log(pi)
    log_det = np.sum(np.log(2.0 * np.pi * sigma), axis=1)
    data_term = (z * z) @ inv_cov.T
    proto_term = np.sum(mu * mu * inv_cov, axis=1)
    cross_term = z @ (mu * inv_cov).T
    quad = data_term + proto_term[None, :] - 2.0 * cross_term
    log_normal = -0.5 * (quad + log_det[None, :])
    log_post = log_pi[None, :] + log_normal
    max_log = np.max(log_post, axis=1, keepdims=True)
    log_norm = max_log + np.log(
        np.sum(np.exp(log_post - max_log), axis=1, keepdims=True)
    )
    return np.exp(log_post - log_norm).astype(np.float32, copy=False)
