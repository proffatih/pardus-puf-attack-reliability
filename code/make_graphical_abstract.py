"""
make_graphical_abstract.py
==========================
Elsevier graphical abstract (landscape, >= 531 x 1328 px, readable at 5 x 13 cm).
Three panels, left to right: the tension, the price, the transfer.
Every number is read from results/*.csv -- nothing is typed by hand.
"""
from __future__ import annotations
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#8a8a8a"

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13.5,
                     "axes.titlesize": 15, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "lines.linewidth": 2.6,
                     "lines.markersize": 8})


def rows(n):
    with open(os.path.join(RES, n), newline="") as f:
        return list(csv.DictReader(f))


def main():
    d = rows("design_rule_n64.csv")
    ks = [int(r["k"]) for r in d]
    ber = [float(r["ber"]) * 100 for r in d]
    acc = [float(r["attack_acc_at_max_budget"]) * 100 for r in d]
    bits = [int(r["resp_bits"]) for r in d]
    nm = rows("transfer_nomogram.csv")
    tl = rows("transfer_law.csv")
    worst = max(float(r["rel_error_pct"]) for r in tl)

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 6.4))

    # ---- panel 1: the tension -------------------------------------------- #
    ax = axes[0]
    ax.plot(ks, acc, marker="o", color=ORANGE)
    ax.set_xlabel("XOR depth $k$")
    ax.set_ylabel("attack accuracy [%]", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_ylim(40, 105)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ks, ber, marker="s", color=BLUE)
    ax2.set_ylabel("bit-error rate [%]", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax.set_title("1.  The tension\nsecurity up, reliability down")

    # ---- panel 2: the price ---------------------------------------------- #
    ax = axes[1]
    cols = [GREY, GREY, GREEN, GREEN, ORANGE, ORANGE]
    ax.bar([str(k) for k in ks], bits, color=cols, edgecolor="black", linewidth=0.6)
    for i, b in enumerate(bits):
        ax.text(i, b + 22, str(b), ha="center", fontsize=12)
    ax.set_ylim(0, max(bits) * 1.24)
    ax.set_xlabel("XOR depth $k$")
    ax.set_ylabel("PUF response bits per 128-bit key")
    ax.set_title("2.  The price\nBER becomes silicon")
    ax.text(0.42, 0.95, "useful range", transform=ax.transAxes, ha="center",
            fontsize=12.5, color=GREEN, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.9))
    ax.text(0.70, 0.78, "no measured\ngain", transform=ax.transAxes, ha="center",
            fontsize=12, color=ORANGE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.9))

    # ---- panel 3: the transfer ------------------------------------------- #
    ax = axes[2]
    p1s = sorted({float(r["p1_measured"]) for r in nm})
    for k, c in ((1, BLUE), (3, GREEN), (6, ORANGE)):
        y = [int(r["resp_bits"]) for r in nm if int(r["k"]) == k]
        ax.plot([p * 100 for p in p1s], y, marker="o", color=c, label=f"$k={k}$")
    ax.set_yscale("log")
    ax.set_xlabel("single-chain BER measured on your silicon [%]")
    ax.set_ylabel("response bits per 128-bit key")
    ax.set_title(f"3.  The transfer\nXOR law holds to {worst:.1f} %,\nso this applies to real devices")
    ax.legend(fontsize=12)

    fig.tight_layout(pad=1.6)
    out = os.path.join(FIG, "graphical_abstract")
    fig.savefig(out + ".png", dpi=140, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    try:
        from PIL import Image
        w, h = Image.open(out + ".png").size
        print(f"  wrote figures/graphical_abstract.png  {w} x {h} px "
              f"(Elsevier min 1328 x 531)  ratio {w/h:.2f}")
    except Exception:
        print("  wrote figures/graphical_abstract.png")


if __name__ == "__main__":
    main()
