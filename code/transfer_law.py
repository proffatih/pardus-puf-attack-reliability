"""
transfer_law.py
===============
Shows that the two-dimensional (sigma, k) reliability surface measured in this
study collapses onto a ONE-PARAMETER family, and turns that collapse into a
silicon-transfer rule.

For a k-XOR Arbiter PUF the response bit flips whenever an odd number of the k
component chains flips. If the component chains are independent with equal flip
probability p1, the classic error-propagation identity gives

        p_k = ( 1 - (1 - 2 p1)^k ) / 2                                    (*)

We verify (*) against the simulated surface over every (n, sigma, k) cell. If it
holds, the practical consequence is immediate and does not depend on the
simulator at all:

    a designer who MEASURES the single-chain BER p1 of their own silicon can
    read off the k-XOR BER from (*), feed it into the fuzzy-extractor cost model
    (ecc_overhead.py), and obtain the key-extraction overhead for their device
    -- without re-running any simulation.

The simulation therefore does not stand in for silicon; it calibrates a law
whose single free parameter is measured on silicon.

Outputs:
  results/transfer_law.csv          per-cell observed vs predicted BER
  results/transfer_nomogram.csv     p1 (measured on silicon) -> p_k -> ECC cost
"""
from __future__ import annotations
import csv, os
import ecc_overhead as ecc

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))


def xor_law(p1: float, k: int) -> float:
    return (1.0 - (1.0 - 2.0 * p1) ** k) / 2.0


def load_mean_ber():
    acc = {}
    with open(os.path.join(RES, "reliability_ber.csv"), newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["n"]), float(r["sigma_noise"]), int(r["k"]))
            acc.setdefault(key, []).append(float(r["ber"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def validate():
    m = load_mean_ber()
    ns = sorted({k[0] for k in m})
    sigmas = sorted({k[1] for k in m})
    ks = sorted({k[2] for k in m})
    rows, worst = [], 0.0
    for n in ns:
        for s in sigmas:
            p1 = m[(n, s, 1)]
            for k in ks:
                obs = m[(n, s, k)]
                pred = xor_law(p1, k)
                rel = abs(obs - pred) / obs * 100.0
                worst = max(worst, rel)
                rows.append(dict(n=n, sigma=s, k=k, ber_observed=round(obs, 6),
                                 ber_predicted=round(pred, 6),
                                 rel_error_pct=round(rel, 2)))
    with open(os.path.join(RES, "transfer_law.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"transfer_law.csv: {len(rows)} cells, worst relative deviation "
          f"{worst:.2f}%")
    return worst


def nomogram(p1_values=(0.005, 0.01, 0.02, 0.03, 0.05, 0.08),
             ks=(1, 2, 3, 4, 5, 6)):
    """Silicon-transfer table: measured single-chain BER -> key-extraction cost."""
    rows = []
    for p1 in p1_values:
        for k in ks:
            pk = xor_law(p1, k)
            cfg = ecc.cheapest_extractor(pk, key_bits=128, p_target=1e-6, rho=1.0)
            rows.append(dict(
                p1_measured=p1, k=k, ber_k=round(pk, 5),
                ecc=(f"rep{cfg['r']}+BCH({cfg['n']},{cfg['k_bch']},{cfg['t']})"
                     f"x{cfg['blocks']}") if cfg else "infeasible",
                resp_bits=cfg["resp_bits"] if cfg else "",
                helper_bits=cfg["helper_bits"] if cfg else ""))
    with open(os.path.join(RES, "transfer_nomogram.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("transfer_nomogram.csv written")
    print(f"{'p1':>7}" + "".join(f"{('k='+str(k)):>12}" for k in ks))
    for p1 in p1_values:
        cells = [r for r in rows if r["p1_measured"] == p1]
        print(f"{p1:>7}" + "".join(f"{str(c['resp_bits']):>12}" for c in cells))


if __name__ == "__main__":
    validate()
    print()
    nomogram()
