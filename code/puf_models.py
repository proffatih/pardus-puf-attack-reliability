"""
puf_models.py
=============
Canonical additive linear delay model of the Arbiter PUF (APUF) and the
k-XOR Arbiter PUF (XOR-APUF), plus a reliability (noise) model.

This is the STANDARD, field-accepted PUF simulator used throughout the PUF
security literature (Lim et al. 2005, IEEE TVLSI; Ruhrmair et al. 2010, ACM CCS;
Becker 2015, CHES). The delay difference of an n-stage arbiter chain is a linear
function of a transformed (parity / feature) representation of the challenge:

    Delta(c) = w^T Phi(c)                                            (1)

where w in R^{n+1} collects the additive stage-delay parameters and Phi(c) in
{-1,+1}^{n+1} is the parity feature vector

    Phi_i(c) = prod_{j=i}^{n-1} (1 - 2 c_j),   i = 0..n-1,  Phi_n = 1.   (2)

The response is r = sign(Delta(c)) mapped to {0,1}. For a k-XOR APUF, k
independent APUFs are evaluated on the same challenge and their single-bit
responses are XOR-combined.

Reliability is modelled by adding i.i.d. Gaussian noise to the *stage* delay
parameters at each evaluation (thermal/voltage noise), which flips a fraction
of responses; the intra-device bit-error-rate (BER) is the mean flip
probability over a challenge set.

No empirical/measured silicon data are used or claimed: all CRPs are generated
by this physically-motivated model.
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#  Feature transform
# --------------------------------------------------------------------------- #
def challenge_to_feature(challenges: np.ndarray) -> np.ndarray:
    """Map 0/1 challenges of shape (N, n) to parity features of shape (N, n+1).

    Phi_i = prod_{j>=i} (1 - 2 c_j), with a trailing constant +1 column.
    Implemented via a reverse cumulative product for O(N n) cost.
    """
    challenges = np.asarray(challenges)
    # d_j = 1 - 2 c_j  in {+1,-1}
    d = 1 - 2 * challenges.astype(np.int8)
    # reverse cumulative product over the stage axis
    rev = np.cumprod(d[:, ::-1], axis=1)[:, ::-1]  # rev[:,i] = prod_{j>=i} d_j
    ones = np.ones((challenges.shape[0], 1), dtype=rev.dtype)
    phi = np.concatenate([rev, ones], axis=1).astype(np.float64)
    return phi


# --------------------------------------------------------------------------- #
#  Arbiter PUF
# --------------------------------------------------------------------------- #
class ArbiterPUF:
    """Single additive-delay Arbiter PUF.

    The weight vector w in R^{n+1} is drawn once (the device "fingerprint").
    Stage delays are Gaussian; the standard parameterisation draws each
    additive weight i.i.d. N(0, 1) (Ruhrmair et al. 2010).
    """

    def __init__(self, n_stages: int, rng: np.random.Generator,
                 sigma_weight: float = 1.0):
        self.n = n_stages
        self.sigma_weight = sigma_weight
        # n stage weights + 1 bias term
        self.w = rng.normal(0.0, sigma_weight, size=n_stages + 1)

    def delay_diff(self, phi: np.ndarray) -> np.ndarray:
        return phi @ self.w

    def response(self, challenges: np.ndarray) -> np.ndarray:
        phi = challenge_to_feature(challenges)
        return (self.delay_diff(phi) > 0).astype(np.int8)

    def noisy_response(self, challenges: np.ndarray, sigma_noise: float,
                       rng: np.random.Generator) -> np.ndarray:
        """Response with i.i.d. Gaussian noise added to each stage weight.

        Models thermal/voltage fluctuation of the additive stage delays.
        """
        phi = challenge_to_feature(challenges)
        noise = rng.normal(0.0, sigma_noise, size=self.w.shape)
        w_noisy = self.w + noise
        return (phi @ w_noisy > 0).astype(np.int8)


# --------------------------------------------------------------------------- #
#  k-XOR Arbiter PUF
# --------------------------------------------------------------------------- #
class XORArbiterPUF:
    """k-XOR Arbiter PUF: XOR of k independent APUF response bits."""

    def __init__(self, n_stages: int, k: int, rng: np.random.Generator,
                 sigma_weight: float = 1.0):
        self.n = n_stages
        self.k = k
        self.apufs = [ArbiterPUF(n_stages, rng, sigma_weight) for _ in range(k)]

    def response(self, challenges: np.ndarray) -> np.ndarray:
        phi = challenge_to_feature(challenges)
        out = np.zeros(challenges.shape[0], dtype=np.int8)
        for a in self.apufs:
            out ^= (phi @ a.w > 0).astype(np.int8)
        return out

    def noisy_response(self, challenges: np.ndarray, sigma_noise: float,
                       rng: np.random.Generator) -> np.ndarray:
        phi = challenge_to_feature(challenges)
        out = np.zeros(challenges.shape[0], dtype=np.int8)
        for a in self.apufs:
            noise = rng.normal(0.0, sigma_noise, size=a.w.shape)
            out ^= (phi @ (a.w + noise) > 0).astype(np.int8)
        return out


# --------------------------------------------------------------------------- #
#  CRP generation
# --------------------------------------------------------------------------- #
def random_challenges(n_crp: int, n_stages: int,
                      rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(n_crp, n_stages), dtype=np.int8)


def generate_crps(puf, n_crp: int, n_stages: int, rng: np.random.Generator):
    """Return (challenges, responses) for a noiseless (golden) read-out."""
    c = random_challenges(n_crp, n_stages, rng)
    r = puf.response(c)
    return c, r


# --------------------------------------------------------------------------- #
#  Reliability / BER
# --------------------------------------------------------------------------- #
def bit_error_rate(puf, challenges: np.ndarray, sigma_noise: float,
                   n_repeats: int, rng: np.random.Generator) -> float:
    """Intra-device BER: mean fraction of responses that differ from the
    golden (noiseless) response across `n_repeats` noisy read-outs.
    """
    golden = puf.response(challenges)
    flips = np.zeros(challenges.shape[0], dtype=np.float64)
    for _ in range(n_repeats):
        noisy = puf.noisy_response(challenges, sigma_noise, rng)
        flips += (noisy != golden)
    return float(np.mean(flips / n_repeats))
