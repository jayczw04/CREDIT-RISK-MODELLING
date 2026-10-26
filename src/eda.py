import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


## default rate + loan count per band 
def default_rate_table(df, col, target="target", q=5, min_share=0.02):
    s = df[col]
    numeric = pd.api.types.is_numeric_dtype(s)

    if numeric:
        cut = pd.qcut(s, q=q, duplicates="drop")
        order = [str(c) for c in cut.cat.categories] + ["Missing"]
        bands = cut.astype(str)
    else:
        s = s.astype(str)
        share = s.value_counts(normalize=True)
        small = share[share < min_share].index
        bands = s.where(~s.isin(small), "Other (small)")

    bands = bands.where(df[col].notna(), "Missing")
    table = df.groupby(bands, observed=True)[target].agg(loans="size", default_rate="mean")

    if numeric:
        table = table.reindex([o for o in order if o in table.index])
    else:
        table = table.sort_values("default_rate")
    return table


## rough strength ranking: worst band rate ÷ best band rate
def rank_variables(df, cols, target="target"):
    rows = []
    for col in cols:
        t = default_rate_table(df, col, target)
        rows.append({
            "variable": col,
            "best_band_rate": t["default_rate"].min(),
            "worst_band_rate": t["default_rate"].max(),
            "spread_x": t["default_rate"].max() / t["default_rate"].min(),
        })
    return pd.DataFrame(rows).sort_values("spread_x", ascending=False).round(3)


## grid of default-rate charts, one per variable
def plot_default_rates(df, cols, target="target", ncols=3, save_path=None):
    avg = df[target].mean()
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), sharey=True)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        t = default_rate_table(df, col, target)
        bars = ax.bar(t.index.astype(str), t["default_rate"], color="#2F6690")
        ax.bar_label(bars, labels=[f"{n:,}" for n in t["loans"]], padding=2, fontsize=7, color="gray")
        ax.axhline(avg, linestyle="--", color="gray", linewidth=1)
        ax.set_title(col, loc="left", fontsize=11)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(f"Default rate by band (dashed line = average {avg:.1%}, labels = loan counts)",
                 x=0.01, ha="left", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()