"""
constrained_node.py
===================
Expresses the key-extraction budget of transfer_law.py in the two currencies a
constrained endpoint actually rations:

  * helper-data storage, held in non-volatile memory, compared against the
    Class-1 and Class-2 device budgets of RFC 7228 (~100 KiB and ~250 KiB of
    code space, respectively);
  * PUF evaluations per 128-bit key reconstruction, i.e. the response bits the
    extractor consumes -- each one a challenge applied and an arbiter latched,
    so this is the quantity that lands on reconstruction latency and on energy
    per authentication.

Produces the table of Section "Cost on a constrained node".
"""
from __future__ import annotations
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))

KIB = 1024
CLASS1_CODE = 100 * KIB       # RFC 7228 Class 1: ~100 KiB code space
CLASS2_CODE = 250 * KIB       # RFC 7228 Class 2: ~250 KiB code space
P1_SHOWN = (0.05, 0.08)       # the two highest measured single-chain error rates
K_SHOWN = (1, 3, 4, 6)


def main():
    src = os.path.join(RES, "transfer_nomogram.csv")
    rows = []
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            p1, k = float(r["p1_measured"]), int(r["k"])
            if k not in K_SHOWN or not any(abs(p1 - p) < 1e-9 for p in P1_SHOWN):
                continue
            helper_bits = int(r["helper_bits"])
            helper_bytes = helper_bits / 8.0
            rows.append(dict(
                p1_measured=p1, k=k,
                evaluations=int(r["resp_bits"]),
                helper_bits=helper_bits,
                helper_bytes=round(helper_bytes, 1),
                pct_class1=round(100.0 * helper_bytes / CLASS1_CODE, 2),
                pct_class2=round(100.0 * helper_bytes / CLASS2_CODE, 2)))
    rows.sort(key=lambda r: (r["p1_measured"], r["k"]))
    out = os.path.join(RES, "constrained_node.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"wrote {out}")
    print(f"{'p1':>5} {'k':>2} {'evals':>8} {'helper B':>9} {'% C1':>6} {'% C2':>6}")
    for r in rows:
        print(f"{r['p1_measured']*100:>4.0f}% {r['k']:>2} {r['evaluations']:>8} "
              f"{r['helper_bytes']:>9.0f} {r['pct_class1']:>6.2f} {r['pct_class2']:>6.2f}")
    worst = max(rows, key=lambda r: r["helper_bytes"])
    base = next(r for r in rows if r["k"] == 1 and abs(r["p1_measured"] - 0.05) < 1e-9)
    print(f"\nworst helper-data footprint: {worst['helper_bytes']:.0f} B "
          f"= {worst['pct_class1']:.2f}% of a Class-1 code budget")
    print(f"evaluation growth vs (k=1, p1=5%): "
          f"{max(r['evaluations'] for r in rows) / base['evaluations']:.0f}x")


if __name__ == "__main__":
    main()
