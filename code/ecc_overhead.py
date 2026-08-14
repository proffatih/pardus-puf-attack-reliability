"""
ecc_overhead.py
===============
Turns the measured intra-device bit-error rate (BER) of an APUF / k-XOR APUF
into the *silicon cost* a deployed system actually pays: the number of PUF
response bits and the amount of helper data a code-offset fuzzy extractor needs
in order to reconstruct a 128-bit key with failure probability at most
P_target.

This closes the loop of the security/reliability trade-off: XORing k chains buys
modeling-attack resistance (Section V) and is paid for in error-correction
overhead, quantified here.

Construction (standard, cf. Maes et al. CHES 2009; Delvaux et al. CHES 2016):

  repetition(r)  ->  majority vote      (reduces p to p_eff)
  BCH(n, kk, t)  ->  code-offset syndrome helper data

Per block:
    response bits consumed :  r * n
    helper bits published  :  (n - kk)          [BCH syndrome]
                            + n * (r - 1)       [repetition helper]
    secrecy contributed    :  r*n*rho - helper_bits
    block failure          :  P(> t errors in n bits at p_eff)

with rho the min-entropy per PUF response bit (rho = 1 is the ideal-source
upper bound on efficiency; rho < 1 is the honest, biased-source case).

BCH dimensions are NOT taken from a hard-coded table: for a primitive
narrow-sense BCH code of length n = 2^m - 1 the dimension is
    kk = n - |C_1 u C_3 u ... u C_{2t-1}|
where C_j is the 2-cyclotomic coset of j modulo n. Computed exactly below, so
every (n, kk, t) triple in the paper is reproducible from first principles.
"""

from __future__ import annotations
import csv, math, os
from functools import lru_cache

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# --------------------------------------------------------------------------- #
#  BCH parameters from cyclotomic cosets
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def bch_codes(m: int):
    """All narrow-sense primitive BCH codes of length n = 2^m - 1.

    Returns a list of (n, kk, t) with kk > 0, t increasing.
    """
    n = 2 ** m - 1
    cosets, seen = {}, set()
    for j in range(1, n, 2):            # odd representatives suffice
        if j in seen:
            continue
        c, x = set(), j
        while x not in c:
            c.add(x)
            x = (2 * x) % n
        cosets[j] = c
        seen |= c
    out, union = [], set()
    for t in range(1, (n - 1) // 2 + 1):
        j = 2 * t - 1
        # union of C_1, C_3, ..., C_{2t-1}; a coset may already be included
        rep = j
        while rep not in cosets and rep > 0:      # find the odd representative
            rep -= 2
        for jj in range(1, j + 1, 2):
            if jj in cosets:
                union |= cosets[jj]
        kk = n - len(union)
        if kk <= 0:
            break
        if not out or out[-1][1] != kk:           # keep only distinct dimensions
            out.append((n, kk, t))
    return out


# --------------------------------------------------------------------------- #
#  Failure probabilities
# --------------------------------------------------------------------------- #
def binom_tail_gt(n: int, t: int, p: float) -> float:
    """P(more than t successes in n Bernoulli(p) trials), computed exactly."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    # sum the lower tail then complement, in log space for stability
    acc = 0.0
    for i in range(0, t + 1):
        acc += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return max(0.0, 1.0 - acc)


def repetition_p_eff(p: float, r: int) -> float:
    """Residual error rate after majority voting over r (odd) repetitions."""
    if r == 1:
        return p
    return binom_tail_gt(r, r // 2, p)


# --------------------------------------------------------------------------- #
#  Design-space search
# --------------------------------------------------------------------------- #
def cheapest_extractor(p: float, key_bits: int = 128,
                       p_target: float = 1e-6, rho: float = 1.0,
                       m_range=(6, 7, 8, 9), r_range=(1, 3, 5, 7, 9, 11)):
    """Minimum-response-bit code-offset fuzzy extractor for BER p.

    Returns a dict describing the cheapest admissible (m, t, r) configuration,
    or None if no configuration in the search space meets the constraints.
    """
    best = None
    for r in r_range:
        p_eff = repetition_p_eff(p, r)
        for m in m_range:
            for (n, kk, t) in bch_codes(m):
                helper = (n - kk) + n * (r - 1)
                resp = r * n
                secrecy = resp * rho - helper
                if secrecy <= 0:
                    continue
                blocks = math.ceil(key_bits / secrecy)
                p_block = binom_tail_gt(n, t, p_eff)
                # total failure over the blocks the key needs
                p_fail = 1.0 - (1.0 - p_block) ** blocks
                if p_fail > p_target:
                    continue
                total_resp = blocks * resp
                total_help = blocks * helper
                cand = dict(ber=p, r=r, n=n, k_bch=kk, t=t, blocks=blocks,
                            resp_bits=total_resp, helper_bits=total_help,
                            p_block=p_block, p_fail=p_fail,
                            bits_per_key_bit=total_resp / key_bits)
                if best is None or cand["resp_bits"] < best["resp_bits"]:
                    best = cand
    return best


# --------------------------------------------------------------------------- #
#  Drive it from the measured BER table
# --------------------------------------------------------------------------- #
def mean_ber_table(path):
    """Average the per-seed BER measurements over seeds."""
    acc = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["n"]), int(row["k"]), float(row["sigma_noise"]))
            acc.setdefault(key, []).append(float(row["ber"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    src = os.path.join(RESULTS, "reliability_ber.csv")
    table = mean_ber_table(src)
    rows = []
    for (n_stages, k, sigma), ber in sorted(table.items()):
        for rho in (1.0, 0.9):
            cfg = cheapest_extractor(ber, key_bits=128, p_target=1e-6, rho=rho)
            if cfg is None:
                rows.append(dict(n=n_stages, k=k, sigma=sigma, ber=ber, rho=rho,
                                 feasible=0, r="", n_bch="", k_bch="", t="",
                                 blocks="", resp_bits="", helper_bits="",
                                 bits_per_key_bit="", p_fail=""))
                continue
            rows.append(dict(n=n_stages, k=k, sigma=sigma, ber=round(ber, 6),
                             rho=rho, feasible=1, r=cfg["r"], n_bch=cfg["n"],
                             k_bch=cfg["k_bch"], t=cfg["t"], blocks=cfg["blocks"],
                             resp_bits=cfg["resp_bits"],
                             helper_bits=cfg["helper_bits"],
                             bits_per_key_bit=round(cfg["bits_per_key_bit"], 2),
                             p_fail=f"{cfg['p_fail']:.3e}"))
    out = os.path.join(RESULTS, "ecc_overhead.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    for r in rows:
        if r["rho"] == 1.0:
            print(f"  n={r['n']} k={r['k']} sigma={r['sigma']} BER={r['ber']} "
                  f"-> rep{r['r']} BCH({r['n_bch']},{r['k_bch']},{r['t']}) "
                  f"x{r['blocks']} = {r['resp_bits']} response bits, "
                  f"{r['helper_bits']} helper bits")


if __name__ == "__main__":
    main()
