"""Why the app moved from `heard` to EffNet, in numbers and two charts.

Every rated pair from listening rounds one and two is scored in three spaces —
the original 518 FMA columns (`all`), the 154-column `heard` space, and the
1,280-dim Discogs-EffNet embedding — and each space's distances are compared
with the 1-5 scores listeners gave blind.

    uv run python -m modeling.effnet_vs_heard

Writes to reports/effnet_vs_heard/:

    agreement.csv      Spearman per round x subset x space, with bootstrap 95% CI
    thirds.csv         mean listener score by distance third, top-five pairs only
    fig1_agreement.png how closely each space agrees with listeners
    fig2_thirds.png    does "closer" in the space mean "more alike" to the ear?

Read this before quoting it: the pairs were picked by `all` and `heard`, not by
EffNet. What is tested is EffNet's sense of distance on pairs people rated, not
EffNet's own recommendations — nobody has listened to those yet. The analysis
was run after the ratings came in; it was not pre-registered.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from modeling.data import load_fma
from modeling.embedding_vs_ear import distances, load_npz, scored_pairs
from modeling.features import space
from modeling.listening import responses

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
EFFNET = ROOT / "data" / "effnet_embeddings_full.npz"
OUT = ROOT / "reports" / "effnet_vs_heard"
TOP = {"model", "heard_top", "all_top"}  # each round's "top five" pairs
SPACES = ["all", "heard", "effnet"]
LABEL = {"all": "Original 518 features", "heard": "heard (154 features)",
         "effnet": "EffNet (pretrained)"}
# Reference palette: EffNet is the subject (orange), heard the comparison
# (blue), the original space is context and stays gray.
COLOR = {"all": "#a9a8a2", "heard": "#2a78d6", "effnet": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
N_BOOT = 4000


def build_pairs() -> pd.DataFrame:
    pairs = scored_pairs(responses())
    # P = round one, Q = round two. Later rounds (T = four) run on fma_medium,
    # where `heard` does not exist, so they are left out here.
    pairs = pairs[pairs.index.str[0].isin(["P", "Q"])]
    pairs["round"] = np.where(pairs.index.str.startswith("Q"), 2, 1)
    X, _ = load_fma("small")
    spaces = {n: space(n, X) for n in ["all", "heard"]}
    spaces["effnet"] = load_npz(EFFNET)
    covered = set.intersection(*(set(S.index) for S in spaces.values()))
    keep = pairs.a.isin(covered) & pairs.b.isin(covered)
    if (~keep).any():
        logger.warning("%s pairs dropped: not every space covers both tracks", (~keep).sum())
    pairs = pairs[keep].copy()
    for n, S in spaces.items():
        Z = pd.DataFrame(StandardScaler().fit_transform(S), index=S.index)
        pairs[n] = distances(Z, pairs)
    return pairs


def agreement(pairs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Spearman(distance, score) per round and subset, sign flipped so higher is better."""
    rng = np.random.default_rng(seed)
    rows = []
    for rnd in [1, 2]:
        P = pairs[pairs["round"] == rnd]
        for subset, S in [("every rated pair", P), ("top five only", P[P.kind.isin(TOP)])]:
            idx = [rng.integers(0, len(S), len(S)) for _ in range(N_BOOT)]
            for n in SPACES:
                rho, p = spearmanr(S[n], S.human)
                boot = [spearmanr(S[n].values[i], S.human.values[i])[0] for i in idx]
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                rows.append({"round": rnd, "subset": subset, "pairs": len(S), "space": n,
                             "spearman": round(rho, 3), "p": round(p, 4),
                             "agreement": round(-rho, 3),
                             "ci_low": round(-hi, 3), "ci_high": round(-lo, 3)})
    return pd.DataFrame(rows)


def thirds(pairs: pd.DataFrame) -> pd.DataFrame:
    """Top-five pairs split into equal thirds by each space's distance."""
    T = pairs[pairs.kind.isin(TOP)]
    rows = []
    for n in ["heard", "effnet"]:
        g = pd.qcut(T[n], 3, labels=["nearest", "middle", "farthest"])
        for third, S in T.groupby(g, observed=True):
            rows.append({"space": n, "third": third, "pairs": len(S),
                         "mean_score": round(S.human.mean(), 2),
                         "sem": round(S.human.std(ddof=1) / np.sqrt(len(S)), 2)})
    return pd.DataFrame(rows)


def _style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_agreement(a: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    groups = [(1, "every rated pair"), (1, "top five only"),
              (2, "every rated pair"), (2, "top five only")]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    width, gap = 0.26, 0.02
    for j, n in enumerate(SPACES):
        for i, (rnd, sub) in enumerate(groups):
            r = a[(a["round"] == rnd) & (a.subset == sub) & (a.space == n)].iloc[0]
            x = i + (j - 1) * (width + gap)
            ax.bar(x, r.agreement, width, color=COLOR[n], label=LABEL[n] if i == 0 else None,
                   zorder=2)
            ax.errorbar(x, r.agreement, yerr=[[r.agreement - r.ci_low], [r.ci_high - r.agreement]],
                        fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            if n == "effnet":
                ax.text(x, r.ci_high + 0.03, f"{r.agreement:.2f}", ha="center", va="bottom",
                        fontsize=10, color=INK, fontweight="bold")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Round {r}\n{s} (n={a[(a['round']==r)&(a.subset==s)].pairs.iloc[0]})"
                        for r, s in groups], color=INK)
    ax.set_ylabel("Agreement with listeners\n(−Spearman ρ, higher is better)", color=INK2)
    ax.set_ylim(-0.4, 1.1)
    _style(ax)
    ax.set_title("EffNet agrees with what listeners heard — in both rounds, and among the top five",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=10, ncol=3)
    fig.text(0.01, 0.01, "Bars: Spearman between each space's distance and the mean 1–5 listener "
             "score, sign flipped. Whiskers: bootstrap 95% CI. 0 = no relation.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_thirds(t: pd.DataFrame, random_floor: float, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    xs = [0, 1, 2]
    for n in ["heard", "effnet"]:
        r = t[t.space == n]
        ax.errorbar(xs, r["mean_score"].values, yerr=r["sem"].values, color=COLOR[n], linewidth=2, marker="o",
                    markersize=8, markeredgecolor="white", markeredgewidth=2, capsize=0,
                    elinewidth=1, zorder=3)
        ax.text(2.32, r.mean_score.iloc[-1], LABEL[n], va="center", fontsize=10.5, color=INK)
        for x, v in zip(xs, r.mean_score):
            side = -0.07 if n == "effnet" else 0.07
            # heard's line rises then falls, so its labels sit off the line, not on it
            dy = 0 if n == "effnet" else (-0.08 if x == 0 else 0.08)
            ax.text(x + side, v + dy, f"{v:.2f}", ha="right" if side < 0 else "left",
                    va="center", fontsize=9.5, color=INK2)
    ax.axhline(random_floor, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0, random_floor - 0.17, f"random pairs: {random_floor:.2f}", fontsize=9, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["nearest third", "middle third", "farthest third"], color=INK)
    ax.set_xlim(-0.3, 3.1)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean listener score (1–5)", color=INK2)
    ax.set_xlabel(f"Top-five pairs from both rounds (n={t.pairs.sum() // 2}), split by each "
                  "space's own distance", color=INK2, fontsize=9.5)
    _style(ax)
    ax.set_title("Among the top five, only EffNet's “closer” sounds closer",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs()
    logger.info("%s rated pairs: %s", len(pairs), pairs.groupby("round").size().to_dict())
    a, t = agreement(pairs), thirds(pairs)
    floor = pairs.loc[pairs.kind == "random", "human"].mean()
    a.to_csv(OUT / "agreement.csv", index=False)
    t.to_csv(OUT / "thirds.csv", index=False)
    fig_agreement(a, OUT / "fig1_agreement.png")
    fig_thirds(t, floor, OUT / "fig2_thirds.png")
    pd.set_option("display.width", 140)
    print(a.pivot_table(index=["round", "subset", "pairs"], columns="space",
                        values="agreement")[SPACES].to_string())
    print()
    print(t.pivot(index="third", columns="space", values="mean_score").loc[
        ["nearest", "middle", "farthest"]].to_string())
    print(f"\nrandom pairs mean score: {floor:.2f}")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
