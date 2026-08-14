"""
make_figures_ext.py
===================
The two figures added for the design-rule / silicon-transfer sections:

  fig_design_frontier.{pdf,png}  attacker CRP cost vs key-extraction cost, k annotated
  fig_transfer.{pdf,png}         (a) XOR error-propagation law vs simulation
                                 (b) silicon-transfer nomogram: measured p1 -> response bits
"""
from __future__ import annotations
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, '/home/pardus/Papers/_templates')
try:
    from pro_style import apply as _pa; _pa()
except Exception:
    pass

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 110, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.35, "lines.linewidth": 1.9, "lines.markersize": 6.5,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def rows(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf/.png")


# --------------------------------------------------------------------------- #
def fig_design_frontier():
    d = rows("design_rule_n64.csv")
    MAXB = max(int(r["max_budget"]) for r in d)
    ks = [int(r["k"]) for r in d]
    resp = [int(r["resp_bits"]) for r in d]
    cost, reached = [], []
    for r in d:
        v = r["crp_to_90pct"]
        if v == "not reached":
            cost.append(MAXB); reached.append(False)
        else:
            cost.append(int(v)); reached.append(True)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(cost, resp, color="0.6", lw=1.4, zorder=1)
    # label offsets tuned per point so that k=4 and k=5 (same x, near-equal y)
    # do not collide
    offs = {1: (10, -14), 2: (10, 8), 3: (12, 4), 4: (14, -16), 5: (14, 8), 6: (14, 2)}
    for x, y, k, ok in zip(cost, resp, ks, reached):
        ax.scatter([x], [y], s=115, zorder=3,
                   color=OKABE[(k - 1) % len(OKABE)],
                   marker="o" if ok else ">",
                   edgecolor="black", linewidth=0.7)
        ax.annotate(f"$k={k}$", (x, y), textcoords="offset points",
                    xytext=offs.get(k, (10, 6)), fontsize=11.5)
    ax.set_xscale("log")
    ax.set_xlim(6e2, 6.5e5)
    ax.set_xlabel("attacker CRP budget to reach 90 % modeling accuracy\n"
                  r"(right-pointing markers: not reached within $2\times10^{5}$ CRPs)")
    ax.set_ylabel("PUF response bits for a 128-bit key\n"
                  r"($P_{\mathrm{fail}}\leq 10^{-6}$)")
    ax.set_title(r"Security bought vs. reliability paid ($n=64$, $\sigma=0.05$)")
    ax.annotate("knee at $k=3$:\n$50\\times$ the attacker's data cost\nfor $2\\times$ the silicon cost",
                xy=(cost[2], resp[2]), xytext=(1.15e3, 880),
                fontsize=10.5, ha="left",
                arrowprops=dict(arrowstyle="->", lw=1.2, color="0.35",
                                connectionstyle="arc3,rad=-0.15"))
    save(fig, "fig_design_frontier")


# --------------------------------------------------------------------------- #
def fig_transfer():
    tl = rows("transfer_law.csv")
    nm = rows("transfer_nomogram.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    ax = axes[0]
    obs = np.array([float(r["ber_observed"]) for r in tl])
    pred = np.array([float(r["ber_predicted"]) for r in tl])
    kk = np.array([int(r["k"]) for r in tl])
    lim = [0, max(obs.max(), pred.max()) * 1.07]
    ax.plot(lim, lim, color="0.4", ls="--", lw=1.3, zorder=1,
            label="$p_k=(1-(1-2p_1)^k)/2$")
    for k in sorted(set(kk)):
        s = kk == k
        ax.scatter(pred[s], obs[s], s=52, zorder=3, label=f"$k={k}$",
                   color=OKABE[(k - 1) % len(OKABE)],
                   edgecolor="black", linewidth=0.5)
    worst = max(float(r["rel_error_pct"]) for r in tl)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("BER predicted by the error-propagation law")
    ax.set_ylabel("BER measured in simulation")
    ax.set_title(f"(a) The surface collapses to one parameter\n"
                 f"36 cells, worst deviation {worst:.1f} %")
    ax.legend(ncol=2, fontsize=9, loc="upper left")

    ax = axes[1]
    p1s = sorted({float(r["p1_measured"]) for r in nm})
    for k in sorted({int(r["k"]) for r in nm}):
        y = [int(r["resp_bits"]) for r in nm if int(r["k"]) == k]
        ax.plot([p * 100 for p in p1s], y, marker="o",
                color=OKABE[(k - 1) % len(OKABE)], label=f"$k={k}$")
    ax.set_yscale("log")
    ax.set_xlabel("single-chain BER $p_1$ measured on silicon  [%]")
    ax.set_ylabel("PUF response bits for a 128-bit key")
    ax.set_title("(b) Silicon-transfer nomogram\n"
                 "measure $p_1$ once, read off the budget")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    save(fig, "fig_transfer")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_design_frontier()
    fig_transfer()
