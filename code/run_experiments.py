"""
run_experiments.py
==================
Real ML modeling-attack and reliability experiments on the additive-delay
APUF / k-XOR APUF simulator (see puf_models.py).

Outputs (results/):
  attack_accuracy.csv   -- attack accuracy vs (k, n, train_size, model, seed)
  reliability_ber.csv   -- intra-device BER vs (k, n, sigma_noise, seed)

All numbers are produced by the runs executed here; nothing is hand-set.

Usage:
  python3 run_experiments.py            # full sweep (several minutes on 8 cores)
  python3 run_experiments.py --quick    # reduced sweep for a fast smoke test
"""

from __future__ import annotations
import argparse
import os
import time
import csv
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from puf_models import (XORArbiterPUF, challenge_to_feature,
                        random_challenges, bit_error_rate)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------- #
def run_attack(model_name, phi_tr, y_tr, phi_te, y_te):
    """Fit one attacker and return its test accuracy."""
    if model_name == "LR":
        clf = LogisticRegression(C=10.0, max_iter=3000, solver="lbfgs",
                                 tol=1e-4)
    elif model_name == "MLP":
        # batch size scales with data so large fits stay fast; tanh net is the
        # standard XOR-capable attacker.
        bs = 2048 if len(y_tr) >= 20000 else 256
        clf = MLPClassifier(hidden_layer_sizes=(64, 64),
                            activation="tanh", solver="adam",
                            alpha=1e-4, batch_size=bs,
                            learning_rate_init=3e-3, max_iter=80,
                            early_stopping=True, n_iter_no_change=6,
                            validation_fraction=0.1,
                            random_state=0)
    else:
        raise ValueError(model_name)
    clf.fit(phi_tr, y_tr)
    return float(clf.score(phi_te, y_te))


def attack_sweep(args):
    """Attack accuracy vs k, challenge length n, training size, attacker."""
    out_path = os.path.join(RESULTS, "attack_accuracy.csv")
    n_stages_list = [64] if args.quick else [64, 128]
    k_list = [1, 2, 3, 4] if args.quick else [1, 2, 3, 4, 5, 6]
    if args.quick:
        train_sizes = [1000, 5000, 20000]
    else:
        train_sizes = [1000, 2000, 5000, 10000, 20000,
                       50000, 100000, 200000]
    # LR (fast, exact) is run on the full training-size grid -- it is the
    # primary security curve. The tanh-MLP (XOR-capable nonlinear attacker) is
    # the more costly fit; we run it up to MLP_MAX_TS CRPs, which is already
    # past where its k-dependence is fully resolved.
    MLP_MAX_TS = 50000
    seeds = [0, 1, 2] if args.quick else [0, 1, 2, 3]
    n_test = 20000

    rows = []
    t0 = time.time()
    for n in n_stages_list:
        for k in k_list:
            for seed in seeds:
                rng = np.random.default_rng(1000 * seed + n + k)
                puf = XORArbiterPUF(n, k, rng)
                max_train = max(train_sizes)
                # one large challenge pool per (n,k,seed); subsample train sizes
                c_pool = random_challenges(max_train + n_test, n, rng)
                r_pool = puf.response(c_pool)
                phi_pool = challenge_to_feature(c_pool)
                phi_te = phi_pool[max_train:max_train + n_test]
                y_te = r_pool[max_train:max_train + n_test]
                for ts in train_sizes:
                    phi_tr = phi_pool[:ts]
                    y_tr = r_pool[:ts]
                    models = ["LR", "MLP"] if ts <= MLP_MAX_TS else ["LR"]
                    for m in models:
                        acc = run_attack(m, phi_tr, y_tr, phi_te, y_te)
                        rows.append(dict(n=n, k=k, seed=seed, train_size=ts,
                                         model=m, test_acc=acc, n_test=n_test))
                        print(f"[attack] n={n} k={k} seed={seed} "
                              f"train={ts:>7} {m:<3} acc={acc:.4f} "
                              f"({time.time()-t0:6.1f}s)")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}  ({len(rows)} rows)")


def reliability_sweep(args):
    """Intra-device BER vs k and noise level (the reliability cost of XORing)."""
    out_path = os.path.join(RESULTS, "reliability_ber.csv")
    n_stages_list = [64] if args.quick else [64, 128]
    k_list = [1, 2, 3, 4] if args.quick else [1, 2, 3, 4, 5, 6]
    sigma_list = [0.025, 0.05, 0.1]
    seeds = [0, 1, 2] if args.quick else [0, 1, 2, 3, 4]
    n_eval = 10000
    n_repeats = 21

    rows = []
    t0 = time.time()
    for n in n_stages_list:
        for k in k_list:
            for sigma in sigma_list:
                for seed in seeds:
                    rng = np.random.default_rng(7000 * seed + n + k)
                    puf = XORArbiterPUF(n, k, rng)
                    c = random_challenges(n_eval, n, rng)
                    ber = bit_error_rate(puf, c, sigma, n_repeats, rng)
                    rows.append(dict(n=n, k=k, sigma_noise=sigma, seed=seed,
                                     ber=ber, n_eval=n_eval,
                                     n_repeats=n_repeats))
                    print(f"[relia ] n={n} k={k} sigma={sigma} seed={seed} "
                          f"BER={ber:.4f} ({time.time()-t0:6.1f}s)")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["attack", "reliability"], default=None)
    args = ap.parse_args()
    if args.only in (None, "reliability"):
        reliability_sweep(args)
    if args.only in (None, "attack"):
        attack_sweep(args)
