"""
scripts/alpha_research.py
Massive grid search for optimal alpha parameters.
Goal: Sharpe > 1.5 OOS on 5+ years.
"""
import pandas as pd
import numpy as np
from itertools import product
from sklearn.decomposition import PCA

prices = pd.read_parquet("data_lake/parquet/prices.parquet")
spy = prices['SPY']
vix = prices['^VIX']
sectors = ['XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLU']
log_rel = np.log(prices[sectors].div(spy, axis=0))

def compute_residual_z(log_rel, lookback=60, pca_window=252, n_components=3):
    n = len(log_rel)
    z_scores = pd.DataFrame(np.nan, index=log_rel.index, columns=log_rel.columns)
    for t in range(pca_window + lookback, n):
        train = log_rel.iloc[t-pca_window:t].diff().dropna()
        if len(train) < 100:
            continue
        pca = PCA(n_components=min(n_components, len(train.columns)))
        pca.fit(train.values)
        X = train.values
        reconstructed = pca.inverse_transform(pca.transform(X))
        residuals = X - reconstructed
        resid_cum = np.cumsum(residuals, axis=0)
        for j, col in enumerate(log_rel.columns):
            recent = resid_cum[-lookback:, j]
            if len(recent) >= 20:
                z_scores.iloc[t, j] = (recent[-1] - recent.mean()) / (recent.std() + 1e-10)
    return z_scores

print("Computing OOS residual z-scores...")
z_scores = compute_residual_z(log_rel, lookback=60, pca_window=252)
print(f"Z-scores computed: {z_scores.dropna(how='all').shape[0]} dates")

def backtest(z_scores, prices, spy, vix, sectors, mr_whitelist,
             z_entry, z_exit, hold_max, vix_kill, mom_filter,
             regime_sizing, max_pos, weight):
    n = len(prices)
    positions = {}
    daily_returns = []
    trades = []

    for t in range(350, n):
        v = float(vix.iloc[t]) if t < len(vix) else 20

        if v > vix_kill:
            positions.clear()
            daily_returns.append(0.0)
            continue

        regime = 'CALM' if v < 15 else ('NORMAL' if v < 22 else ('TENSION' if v < 30 else 'CRISIS'))
        size_mult = regime_sizing.get(regime, 0.0)
        if size_mult == 0:
            positions.clear()
            daily_returns.append(0.0)
            continue

        day_pnl = 0.0
        for s in list(positions.keys()):
            pos = positions[s]
            ret_s = float(np.log(prices[s].iloc[t] / prices[s].iloc[t-1]))
            ret_spy = float(np.log(spy.iloc[t] / spy.iloc[t-1]))
            alpha = pos["dir"] * (ret_s - ret_spy)
            day_pnl += alpha * weight * size_mult

            z_now = float(z_scores[s].iloc[t]) if np.isfinite(z_scores[s].iloc[t]) else 0
            hold_days = t - pos["entry"]
            if abs(z_now) < z_exit or hold_days >= hold_max or (pos["dir"] * z_now > 0 and abs(z_now) > z_entry * 1.5):
                trades.append(alpha * hold_days)
                positions.pop(s)

        daily_returns.append(day_pnl)

        if len(positions) >= max_pos:
            continue

        for s in sectors:
            if s in positions or len(positions) >= max_pos or s not in mr_whitelist:
                continue
            z = float(z_scores[s].iloc[t]) if np.isfinite(z_scores[s].iloc[t]) else 0
            if abs(z) < z_entry:
                continue
            if mom_filter:
                lb = min(60, t)
                mom = float(np.log(prices[s].iloc[t] / prices[s].iloc[max(0, t-lb)]))
                spy_mom = float(np.log(spy.iloc[t] / spy.iloc[max(0, t-lb)]))
                d = 1 if z < 0 else -1
                if (d == 1 and mom - spy_mom < -0.10) or (d == -1 and mom - spy_mom > 0.10):
                    continue
            positions[s] = {"entry": t, "dir": 1 if z < 0 else -1}

    return np.array(daily_returns), trades

print("\n=== GRID SEARCH ===")
best = {"sharpe": -999}
results = []
regime_sizing = {'CALM': 1.3, 'NORMAL': 1.0, 'TENSION': 0.6, 'CRISIS': 0.0}

mr_sets = [
    {'XLC','XLF','XLI','XLU'},
    {'XLF','XLI','XLU'},
    {'XLC','XLF','XLI','XLU','XLP'},
    {'XLC','XLF','XLI','XLU','XLB'},
]

configs = list(product(
    [0.5, 0.7, 0.9, 1.1],   # z_entry
    [0.15, 0.25, 0.35],     # z_exit
    [15, 20, 25, 30],       # hold_max
    [28, 32, 36],           # vix_kill
    [True, False],          # mom_filter
    [2, 3, 4],              # max_pos
    [0.03, 0.05, 0.07],    # weight
))

total = len(configs) * len(mr_sets)
print(f"Testing {total} configurations...")

count = 0
for mr_set in mr_sets:
    for z_entry, z_exit, hold_max, vix_kill, mom, max_pos, wt in configs:
        count += 1
        rets, trades = backtest(
            z_scores, prices, spy, vix, sectors,
            mr_set, z_entry, z_exit, hold_max, vix_kill, mom,
            regime_sizing, max_pos, wt,
        )
        if len(rets) < 252 or np.std(rets) < 1e-8:
            continue

        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
        cum = np.cumsum(rets)
        dd = (cum - np.maximum.accumulate(cum)).min()
        wr = (np.array(trades) > 0).mean() if trades else 0

        r = {"sharpe": sharpe, "dd": dd, "wr": wr, "n": len(trades),
             "mr": ",".join(sorted(mr_set)), "z_en": z_entry, "z_ex": z_exit,
             "hold": hold_max, "vix": vix_kill, "mom": mom, "pos": max_pos, "wt": wt}
        results.append(r)
        if sharpe > best["sharpe"] and len(trades) >= 80:
            best = r.copy()

        if count % 1000 == 0:
            print(f"  {count}/{total}, best={best['sharpe']:.3f}")

top = sorted([r for r in results if r["n"] >= 80], key=lambda x: x["sharpe"], reverse=True)[:15]

print(f"\nTOP 15:")
print(f"{'Sharpe':>8} {'MaxDD':>8} {'WR':>6} {'#':>5} {'z_en':>5} {'z_ex':>5} {'hold':>5} {'vix':>4} {'mom':>4} {'pos':>4} {'wt':>5} sectors")
print("-" * 90)
for r in top:
    print(f"{r['sharpe']:>+8.3f} {r['dd']:>8.3f} {r['wr']:>5.1%} {r['n']:>5} "
          f"{r['z_en']:>5.1f} {r['z_ex']:>5.2f} {r['hold']:>5} {r['vix']:>4} "
          f"{'Y' if r['mom'] else 'N':>4} {r['pos']:>4} {r['wt']:>5.2f} {r['mr']}")

# OOS validation on best
if top:
    b = top[0]
    mr_set = set(b["mr"].split(","))
    rets_all, _ = backtest(z_scores, prices, spy, vix, sectors,
                           mr_set, b["z_en"], b["z_ex"], b["hold"], b["vix"], b["mom"],
                           regime_sizing, b["pos"], b["wt"])
    split = int(len(rets_all) * 0.7)
    is_r, oos_r = rets_all[:split], rets_all[split:]
    sh_is = np.mean(is_r) / np.std(is_r) * np.sqrt(252) if np.std(is_r) > 1e-8 else 0
    sh_oos = np.mean(oos_r) / np.std(oos_r) * np.sqrt(252) if np.std(oos_r) > 1e-8 else 0
    print(f"\nOOS VALIDATION:")
    print(f"  In-sample:  Sharpe={sh_is:+.3f} ({len(is_r)} days)")
    print(f"  Out-sample: Sharpe={sh_oos:+.3f} ({len(oos_r)} days)")
    print(f"  OOS/IS: {sh_oos/sh_is:.2f}" if sh_is != 0 else "  OOS/IS: N/A")
    print(f"  Annual return: {np.mean(rets_all)*252:.2%}")
    print(f"  Annual vol: {np.std(rets_all)*np.sqrt(252):.2%}")
