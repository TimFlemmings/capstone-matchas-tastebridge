"""Answer round four's pre-registered questions, exactly as written beforehand.

    uv run python -m modeling.round4_analysis

Reads every `R*d_ratings.csv` in data/listening_test/responses/ (put only the
files that arrived before the deadline there; later ones are reported apart),
joins them to the key in data/listening_test/round4/, and writes
reports/round4/: per-pair scores, the three answers, and one chart.

The questions, tests and thresholds are those in
data/listening_test/ROUND4_PREREGISTRATION.md — nothing here was chosen after
seeing a rating.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from modeling.round2 import TEST_DIR

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
KEY = TEST_DIR / "round4" / "pairs_key_r4.csv"
RESPONSES = TEST_DIR / "responses"
OUT = ROOT / "reports" / "round4"
TIERS = ["home", "bridge", "adventure"]
LABEL = {"home": "Home", "bridge": "Bridge", "adventure": "Adventure", "random": "Random"}
POOLED_FLOOR = 1.57   # rounds one and two's random pairs, fixed in the pre-registration
RELATED_MIN = 2.0     # "weak: one or two minor characteristics connect them"
ALPHA = 0.05


def load(responses: Path = RESPONSES, key_path: Path = KEY) -> pd.DataFrame:
    files = sorted(responses.glob("R*d_ratings.csv"))
    if not files:
        raise FileNotFoundError(f"no R*d_ratings.csv in {responses}")
    ratings = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = pd.read_csv(key_path)
    d = ratings.merge(key, on="pair_id", how="inner")
    logger.info("%s ratings from %s raters on %s pairs",
                len(d), d.rater.nunique(), d.pair_id.nunique())
    return d


def per_pair(d: pd.DataFrame) -> pd.DataFrame:
    return (d.dropna(subset=["similarity_1_5"])
            .groupby("pair_id")
            .agg(score=("similarity_1_5", "mean"), votes=("similarity_1_5", "size"),
                 kind=("kind", "first"), seed=("seed", "first"), rank=("rank", "first"),
                 seed_artist=("seed_artist", "first"), other_artist=("other_artist", "first"),
                 other_title=("other_title", "first"))
            .reset_index())


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        adjusted[k] = running
    return adjusted


def answer(p: pd.DataFrame) -> dict:
    rnd = p.loc[p.kind == "random", "score"]
    q1 = {}
    raw = {}
    for t in TIERS:
        s = p.loc[p.kind == t, "score"]
        raw[t] = mannwhitneyu(s, rnd, alternative="greater").pvalue if len(rnd) and len(s) else np.nan
        sec = wilcoxon(s - POOLED_FLOOR, alternative="greater").pvalue if len(s) > 1 else np.nan
        q1[t] = {"pairs": len(s), "mean": s.mean(), "random_mean": rnd.mean(),
                 "p_raw": raw[t], "p_secondary_vs_1.57": sec}
    for t, adj in holm({k: v for k, v in raw.items() if not np.isnan(v)}).items():
        q1[t]["p_holm"] = adj
        q1[t]["beats_random"] = adj < ALPHA

    tiered = p[p.kind.isin(TIERS)]
    rho, p_two = spearmanr(tiered.kind.map({t: i for i, t in enumerate(TIERS, 1)}), tiered.score)
    p_one = p_two / 2 if rho < 0 else 1 - p_two / 2
    q2 = {"rho": rho, "p_one_sided": p_one, "ordered": bool(rho < 0 and p_one < ALPHA)}

    adv = q1["adventure"]
    if adv.get("beats_random") and adv["mean"] >= RELATED_MIN:
        verdict = "still related"
    elif adv.get("beats_random"):
        verdict = "barely related"
    else:
        verdict = "not shown to be related"
    return {"q1": q1, "q2": q2, "q3": verdict}


def chart(p: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    color = {"home": "#2a78d6", "bridge": "#1baf7a", "adventure": "#eb6834", "random": "#a9a8a2"}
    kinds = TIERS + ["random"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    rng = np.random.default_rng(0)
    for i, k in enumerate(kinds):
        s = p.loc[p.kind == k, "score"]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(s)), s, s=40, color=color[k], alpha=0.55,
                   edgecolor="white", linewidth=1, zorder=2)
        if len(s):
            ax.plot([i - 0.25, i + 0.25], [s.mean()] * 2, color=color[k], linewidth=3, zorder=3)
            ax.text(i + 0.3, s.mean(), f"{s.mean():.2f}", va="center", fontsize=10, color=ink)
    ax.axhline(RELATED_MIN, color=ink2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(3.45, RELATED_MIN + 0.06, "2 = weak connection", fontsize=8.5, color=ink2, ha="right")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([LABEL[k] for k in kinds], color=ink)
    ax.set_ylim(0.8, 5.2)
    ax.set_ylabel("Listener score per pair (1–5)", color=ink2)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, length=0)
    ax.yaxis.grid(True, color=grid)
    ax.set_axisbelow(True)
    ax.set_title("What the live app recommends, rated blind — by tier", loc="left",
                 fontsize=12.5, color=ink, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    """A markdown table without the optional tabulate dependency."""
    cols = [LABEL.get(c, c) for c in df.columns]
    rows = ["| seed | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for idx, r in df.iterrows():
        rows.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in r) + " |")
    return "\n".join(rows)


def main(responses: Path = RESPONSES, out: Path = OUT):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out.mkdir(parents=True, exist_ok=True)
    p = per_pair(load(responses))
    a = answer(p)
    p.to_csv(out / "pair_scores.csv", index=False)
    seeds = p.pivot_table(index="seed_artist", columns="kind", values="score")[TIERS + ["random"]]
    seeds.round(2).to_csv(out / "per_seed.csv")
    chart(p, out / "fig_tiers.png")

    lines = ["# Round four — results", "",
             f"{len(p)} pairs rated, {p.votes.sum()} ratings.", "",
             "## Q1 — does each tier beat random?", "",
             "| tier | pairs | mean | random mean | p (Holm) | beats random | p vs 1.57 (secondary) |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for t in TIERS:
        r = a["q1"][t]
        lines.append(f"| {LABEL[t]} | {r['pairs']} | {r['mean']:.2f} | {r['random_mean']:.2f} | "
                     f"{r.get('p_holm', np.nan):.4f} | {'yes' if r.get('beats_random') else 'not shown'} | "
                     f"{r['p_secondary_vs_1.57']:.4f} |")
    q2 = a["q2"]
    lines += ["", "## Q2 — Home ≥ Bridge ≥ Adventure?", "",
              f"Spearman ρ = {q2['rho']:.3f}, one-sided p = {q2['p_one_sided']:.4f} → "
              f"{'the dial is heard as a dial' if q2['ordered'] else 'ordering not detectable at this sample size'}.",
              "", "## Q3 — is Adventure still related?", "", f"**{a['q3']}**", "",
              "## Per seed (mean score)", "", _md(seeds), "",
              "![tiers](fig_tiers.png)"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
