"""
design_rule.py
==============
Joins the two axes measured elsewhere in this study into a single, actionable
parameter-selection table for the k-XOR Arbiter PUF:

  security     : the attacker's CRP cost to reach a target modeling accuracy
                 (from attack_accuracy.csv, best attacker over LR and MLP)
  reliability  : the error-correction overhead a 128-bit key extraction pays
                 at the measured BER (from ecc_overhead.py)

The resulting "cost of security" column -- PUF response bits per decade of
attacker CRP cost -- is what a designer actually needs in order to pick k, and
is the quantity no prior study reports on a common footing.
"""
from __future__ import annotations
import csv, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

TARGET_ACC = 0.90     # "broken" threshold used throughout the paper
SIGMA_REF = 0.05      # reference stage-noise level for the design table


def attacker_cost(n_stages: int):
    """CRP budget at which the best attacker first reaches TARGET_ACC, per k.

    Returns {k: (budget or None, best_acc_at_max_budget, model)}.
    """
    rows = []
    with open(os.path.join(RES, "attack_accuracy.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if int(r["n"]) != n_stages:
                continue
            rows.append((int(r["k"]), int(r["train_size"]), r["model"],
                         float(r["test_acc"])))
    out = {}
    for k in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == k]
        # mean accuracy over seeds for each (budget, model)
        agg = {}
        for _, b, m, a in sub:
            agg.setdefault((b, m), []).append(a)
        agg = {kk: sum(v) / len(v) for kk, v in agg.items()}
        reached = [b for (b, m), a in agg.items() if a >= TARGET_ACC]
        budget = min(reached) if reached else None
        bmax = max(b for (b, m) in agg)
        best_at_max = max(a for (b, m), a in agg.items() if b == bmax)
        best_model = max(((a, m) for (b, m), a in agg.items() if b == bmax))[1]
        out[k] = (budget, best_at_max, best_model, bmax)
    return out


def ecc_lookup(n_stages: int, sigma: float, rho: float = 1.0):
    out = {}
    with open(os.path.join(RES, "ecc_overhead.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if (int(r["n"]) == n_stages and abs(float(r["sigma"]) - sigma) < 1e-9
                    and abs(float(r["rho"]) - rho) < 1e-9 and r["feasible"] == "1"):
                out[int(r["k"])] = r
    return out


def main():
    for n_stages in (64, 128):
        atk = attacker_cost(n_stages)
        ecc = ecc_lookup(n_stages, SIGMA_REF)
        rows = []
        base_resp = None
        for k in sorted(atk):
            if k not in ecc:
                continue
            budget, acc_max, model, bmax = atk[k]
            e = ecc[k]
            resp = int(e["resp_bits"])
            if base_resp is None:
                base_resp = resp
            rows.append(dict(
                n=n_stages, k=k, sigma=SIGMA_REF,
                attack_acc_at_max_budget=round(acc_max, 4),
                best_attacker=model, max_budget=bmax,
                crp_to_90pct=(budget if budget is not None else "not reached"),
                ber=e["ber"],
                ecc_code=f"rep{e['r']}+BCH({e['n_bch']},{e['k_bch']},{e['t']})x{e['blocks']}",
                resp_bits=resp, helper_bits=int(e["helper_bits"]),
                resp_bits_rel_k1=round(resp / base_resp, 2),
                key_fail_prob=e["p_fail"],
            ))
        out = os.path.join(RES, f"design_rule_n{n_stages}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n=== n = {n_stages}, sigma = {SIGMA_REF}, 128-bit key, P_fail <= 1e-6 ===")
        print(f"{'k':>2} {'atk acc':>8} {'CRP->90%':>12} {'BER':>8} "
              f"{'ECC':>28} {'resp bits':>10} {'x k=1':>6}")
        for r in rows:
            print(f"{r['k']:>2} {r['attack_acc_at_max_budget']:>8} "
                  f"{str(r['crp_to_90pct']):>12} {float(r['ber']):>8.4f} "
                  f"{r['ecc_code']:>28} {r['resp_bits']:>10} {r['resp_bits_rel_k1']:>6}")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
