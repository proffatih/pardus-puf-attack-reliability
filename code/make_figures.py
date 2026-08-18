"""
make_figures.py
===============
Build the publication figures from results/*.csv. Saves BOTH .pdf and .png
(dpi>=200) to figures/. Also writes a small results/summary tables CSV used by
the manuscript.

Figures:
  fig_acc_vs_trainsize.{pdf,png}  attack accuracy vs training-set size, per k (n=64)
  fig_acc_vs_k.{pdf,png}          attack accuracy vs XOR count k (large train, n=64,128)
  fig_ber_vs_k.{pdf,png}          reliability BER vs k (the security/reliability trade-off)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0,'/home/pardus/Papers/_templates')
try:
    from pro_style import apply as _pa; _pa()
except Exception as _e:
    pass

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.9,
    "lines.markersize": 5.5,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

att = pd.read_csv(os.path.join(RES, "attack_accuracy.csv"))
rel = pd.read_csv(os.path.join(RES, "reliability_ber.csv"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name + ".{pdf,png}")


# --------------------------------------------------------------------------- #
# Fig 1: attack accuracy vs training size, per k  (LR vs MLP, n=64)
# --------------------------------------------------------------------------- #
def fig_acc_vs_trainsize():
    sub = att[att.n == 64]
    ks = sorted(sub.k.unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, model in zip(axes, ["LR", "MLP"]):
        d = sub[sub.model == model]
        for c, k in zip(cmap, ks):
            g = (d[d.k == k].groupby("train_size")["test_acc"]
                 .agg(["mean", "std"]).reset_index())
            ax.errorbar(g.train_size, g["mean"], yerr=g["std"],
                        marker="o", color=c, capsize=2, label=f"k={k}")
        ax.set_xscale("log")
        ax.set_xlabel("Training-set size (CRPs)")
        ax.set_title(f"{model} attacker")
        ax.axhline(0.5, ls=":", color="gray", lw=1)
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("Test accuracy")
    axes[1].legend(title="XOR count", ncol=2, loc="lower right")
    fig.suptitle("Modeling-attack accuracy vs training-set size "
                 "(64-bit APUF / XOR-APUF)", y=1.02)
    save(fig, "fig_acc_vs_trainsize")


# --------------------------------------------------------------------------- #
# Fig 2: attack accuracy vs k at the largest training budget (n=64 and 128)
# --------------------------------------------------------------------------- #
def fig_acc_vs_k():
    lr_ts = int(att[att.model == "LR"].train_size.max())
    mlp_ts = int(att[att.model == "MLP"].train_size.max())
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    styles = {64: "-o", 128: "--s"}
    colors = {"LR": "tab:blue", "MLP": "tab:red"}
    for n in sorted(att.n.unique()):
        for model in ["LR", "MLP"]:
            ts = lr_ts if model == "LR" else mlp_ts
            d = att[(att.n == n) & (att.model == model) &
                    (att.train_size == ts)]
            g = d.groupby("k")["test_acc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g.k, g["mean"], yerr=g["std"], fmt=styles[n],
                        color=colors[model], capsize=3,
                        label=f"{model} (n={n}, {ts//1000}k CRPs)")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("XOR count $k$")
    ax.set_ylabel("Test accuracy at largest budget")
    ax.set_title("Modeling-attack accuracy vs XOR count")
    ax.set_ylim(0.45, 1.02)
    ax.legend()
    save(fig, "fig_acc_vs_k")


# --------------------------------------------------------------------------- #
# Fig 3: reliability BER vs k  (the cost of XORing)
# --------------------------------------------------------------------------- #
def fig_ber_vs_k():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sub = rel[rel.n == 64]
    sigmas = sorted(sub.sigma_noise.unique())
    cmap = plt.cm.plasma(np.linspace(0.1, 0.8, len(sigmas)))
    for c, s in zip(cmap, sigmas):
        g = (sub[sub.sigma_noise == s].groupby("k")["ber"]
             .agg(["mean", "std"]).reset_index())
        ax.errorbar(g.k, 100 * g["mean"], yerr=100 * g["std"], marker="o",
                    color=c, capsize=3, label=fr"$\sigma_{{noise}}={s}$")
    ax.set_xlabel("XOR count $k$")
    ax.set_ylabel("Intra-device BER (%)")
    ax.set_title("Reliability cost of XORing (64-bit chains)")
    ax.legend(title="Stage-delay noise")
    save(fig, "fig_ber_vs_k")


# --------------------------------------------------------------------------- #
# Fig 4: iPUF modeling resistance vs training budget (from ipuf_attack.csv)
# --------------------------------------------------------------------------- #
def fig_ipuf():
    ip = pd.read_csv(os.path.join(RES, "ipuf_attack.csv"))
    order = ["(1,1)", "(1,2)", "(2,2)"]
    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, len(order)))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for c, conf in zip(cmap, order):
        for model, ls, mk in (("MLP", "-", "o"), ("LR", "--", "s")):
            d = ip[(ip.ipuf == conf) & (ip.model == model)]
            g = d.groupby("train_size")["test_acc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g.train_size, g["mean"], yerr=g["std"], ls=ls, marker=mk,
                        color=c, capsize=3,
                        label=fr"{conf} {model}")
    ax.set_xscale("log")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("Training-set size (CRPs)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Interpose-PUF modeling resistance ($n{=}64$)")
    ax.set_ylim(0.45, 1.10)
    # budgets are a small discrete set: label them explicitly instead of leaving
    # a single decade tick, and keep the legend clear of the (2,2) LR series
    budgets = sorted(ip.train_size.unique())
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{int(b/1000)}k" for b in budgets])
    ax.minorticks_off()
    ax.legend(ncol=3, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), framealpha=1.0, edgecolor="0.7")
    save(fig, "fig_ipuf")


# --------------------------------------------------------------------------- #
# Fig 5: BER heatmap over (sigma, k), fine sigma grid (ber_sigma_k.csv)
# --------------------------------------------------------------------------- #
def fig_ber_heatmap():
    bs = pd.read_csv(os.path.join(RES, "ber_sigma_k.csv"))
    bs = bs[bs.n == 64]
    piv = (bs.groupby(["k", "sigma"])["ber"].mean().reset_index()
           .pivot(index="k", columns="sigma", values="ber") * 100.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{s:g}" for s in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel(r"Stage-delay noise $\sigma_{\mathrm{noise}}$")
    ax.set_ylabel("XOR count $k$")
    ax.set_title("Intra-device BER (%) over $(\\sigma,k)$, $n{=}64$")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center",
                    color="white" if piv.values[i, j] < piv.values.max()*0.6 else "black",
                    fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("BER (%)")
    save(fig, "fig_ber_heatmap")


# --------------------------------------------------------------------------- #
# Fig 6: security/reliability trade-off (Pareto) - attack acc vs BER as k varies
# --------------------------------------------------------------------------- #
def fig_tradeoff():
    # best (strongest) attacker accuracy at largest budget per k, n=64
    mlp_ts = int(att[att.model == "MLP"].train_size.max())
    lr_ts = int(att[att.model == "LR"].train_size.max())
    ber = (rel[(rel.n == 64) & (rel.sigma_noise == 0.05)]
           .groupby("k")["ber"].mean() * 100.0)
    ks = sorted(att[att.n == 64].k.unique())
    best_acc = {}
    for k in ks:
        a_lr = att[(att.n == 64) & (att.k == k) & (att.model == "LR") &
                   (att.train_size == lr_ts)].test_acc.mean()
        a_mlp = att[(att.n == 64) & (att.k == k) & (att.model == "MLP") &
                    (att.train_size == mlp_ts)].test_acc.mean()
        best_acc[k] = max(a_lr, a_mlp) * 100.0
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    xs = [ber[k] for k in ks]
    ys = [best_acc[k] for k in ks]
    ax.plot(xs, ys, "-o", color="#2E5A87")
    for k, x, y in zip(ks, xs, ys):
        ax.annotate(f"$k{{=}}{k}$", (x, y), textcoords="offset points",
                    xytext=(7, 6), fontsize=10)
    ax.axhline(50, ls=":", color="gray", lw=1)
    ax.set_xlabel(r"Intra-device BER (%) at $\sigma_{\mathrm{noise}}{=}0.05$")
    ax.set_ylabel("Strongest-attacker accuracy (%)")
    ax.set_title("Security / reliability trade-off ($n{=}64$)")
    ax.set_ylim(45, 102)
    save(fig, "fig_tradeoff")


# --------------------------------------------------------------------------- #
# Fig 7: BER vs k for n=64 and n=128 (reliability ~ independent of n)
# --------------------------------------------------------------------------- #
def fig_ber_vs_n():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sig = 0.05
    styles = {64: "-o", 128: "--s"}
    colors = {64: "#2E5A87", 128: "#C44E52"}
    for n in [64, 128]:
        g = (rel[(rel.n == n) & (rel.sigma_noise == sig)]
             .groupby("k")["ber"].agg(["mean", "std"]).reset_index())
        ax.errorbar(g.k, 100 * g["mean"], yerr=100 * g["std"], fmt=styles[n],
                    color=colors[n], capsize=3, label=f"n={n}")
    ax.set_xlabel("XOR count $k$")
    ax.set_ylabel("Intra-device BER (%)")
    ax.set_title(fr"BER vs $k$ is independent of $n$ ($\sigma={sig}$)")
    ax.legend(title="Challenge length")
    save(fig, "fig_ber_vs_n")


def write_summary_tables():
    # Table A: accuracy vs k at largest budget, LR & MLP, n=64
    max_ts = att.train_size.max()
    rows = []
    for n in sorted(att.n.unique()):
        for k in sorted(att[att.n == n].k.unique()):
            for model in ["LR", "MLP"]:
                d = att[(att.n == n) & (att.k == k) & (att.model == model) &
                        (att.train_size == max_ts)]
                rows.append(dict(n=n, k=k, model=model,
                                 acc_mean=d.test_acc.mean(),
                                 acc_std=d.test_acc.std()))
    pd.DataFrame(rows).to_csv(
        os.path.join(RES, "summary_acc_vs_k.csv"), index=False)
    # Table B: BER vs k (sigma=0.05), n=64
    b = (rel[(rel.n == 64) & (rel.sigma_noise == 0.05)]
         .groupby("k")["ber"].agg(["mean", "std"]).reset_index())
    b.to_csv(os.path.join(RES, "summary_ber_vs_k.csv"), index=False)
    print("wrote summary CSVs; max train size =", max_ts)


if __name__ == "__main__":
    fig_acc_vs_trainsize()
    fig_acc_vs_k()
    fig_ber_vs_k()
    fig_ipuf()
    fig_ber_heatmap()
    fig_tradeoff()
    fig_ber_vs_n()
    write_summary_tables()
