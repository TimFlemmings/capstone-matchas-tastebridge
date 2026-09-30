"""The tuned Familiar / Bridge / Adventurous tier system, as code.

Extracted from notebook 09 verbatim (round 2's settings) so the API and
notebook 10 share one implementation instead of two copies drifting apart.
Needs only `EffNetRecommender` and `ConnectingFactors` — no Essentia, no
audio I/O, so it runs in the project's normal (3.13) environment.

    from modeling.tiers import explore_tiers
    explore_tiers(rec, cf, tracks, seed_id=2, seed_artist="AWOL")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.connecting_factors import ConnectingFactors
from modeling.effnet_recommender import EffNetRecommender

ALL_FACTORS = ("genre", "genre_lineage", "era", "geography")

# Round 2's tuned defaults from notebook 09 -- the settings that "felt right"
# on the Hip-Hop / Instrumental / Pop catalogue seeds plus the external seed.
DEFAULT_BANDS = {"familiar": (0, 15), "bridge": (30, 150), "adventurous": (500, 2000)}
DEFAULT_GATE_FACTORS = {"Bridge": ("era",), "Adventurous": None}
DEFAULT_MAX_SAME_GENRE_SHARE = {"Bridge": 0.25}


def _genre_top(tracks: pd.DataFrame, track_id):
    return tracks.loc[track_id, ("track", "genre_top")]


def _cap_same_genre(tracks, cf, pool_idx, candidate_ids, seed_id, seed_genre, n, max_share,
                     allowed_set, rng, widen_to=None):
    """Split pool_idx by whether the candidate shares seed_genre. Same-genre
    picks only ever come from pool_idx itself, capped at floor(n * max_share).
    Different-genre picks come from pool_idx first; if that's not enough,
    search further out in rank (up to widen_to, exclusive) for more, applying
    the same allowed_set connection gate to anything found there."""
    pool_idx = list(pool_idx)
    if seed_genre is None or not pool_idx:
        return list(rng.choice(pool_idx, size=min(n, len(pool_idx)), replace=False)) if pool_idx else []

    same = [i for i in pool_idx if _genre_top(tracks, candidate_ids[i]) == seed_genre]
    diff = [i for i in pool_idx if _genre_top(tracks, candidate_ids[i]) != seed_genre]
    max_same = int(np.floor(n * max_share))
    n_same = min(max_same, len(same))
    needed_diff = n - n_same

    if len(diff) < needed_diff and widen_to is not None and pool_idx and widen_to > pool_idx[-1]:
        for j in range(pool_idx[-1] + 1, widen_to):
            if len(diff) >= needed_diff:
                break
            if _genre_top(tracks, candidate_ids[j]) == seed_genre:
                continue
            if allowed_set is not None and seed_id is not None:
                if not (set(cf.factors_between(seed_id, candidate_ids[j])) & allowed_set):
                    continue
            diff.append(j)

    n_diff = min(needed_diff, len(diff))
    if n_same + n_diff < n:
        n_same = min(len(same), n - n_diff)

    chosen = list(rng.choice(same, size=n_same, replace=False)) if n_same else []
    chosen += diff[:n_diff]
    return chosen if chosen else pool_idx[: min(n, len(pool_idx))]


def explore_tiers(rec: EffNetRecommender, cf: ConnectingFactors, tracks: pd.DataFrame,
                   seed_id=None, seed_vec=None, seed_artist=None,
                   familiar=DEFAULT_BANDS["familiar"], bridge=DEFAULT_BANDS["bridge"],
                   adventurous=DEFAULT_BANDS["adventurous"], n=5,
                   gate_factors=None, max_same_genre_share=None, widen_to=None, seed=0):
    """Familiar/Bridge/Adventurous picks for one seed.

    Returns {label: [(track_id, distance, rank, factors), ...]}. Pass either
    `seed_id` (a catalogue track) or `seed_vec` (an already-embedded external
    track, unit-normalised) -- see notebook 09 for how the external case is
    built.
    """
    gate_factors = DEFAULT_GATE_FACTORS if gate_factors is None else gate_factors
    max_same_genre_share = DEFAULT_MAX_SAME_GENRE_SHARE if max_same_genre_share is None else max_same_genre_share
    widen_to = {"Bridge": adventurous[0]} if widen_to is None else widen_to

    if seed_vec is None:
        row = rec._position(seed_id)
        seed_vec = rec.embedding_.iloc[[row]]
    widen_values = [v for v in widen_to.values() if v is not None]
    search_n = max(adventurous[1] + 1, (max(widen_values) + 1) if widen_values else 0, 50)
    distances, idx = rec.index_.kneighbors(seed_vec, n_neighbors=search_n)
    distances, idx = distances[0], idx[0]
    if seed_id is not None:
        distances, idx = distances[1:], idx[1:]
    candidate_ids = rec.track_ids_[idx]

    if seed_artist is not None:
        artists = tracks.loc[candidate_ids, ("artist", "name")].to_numpy()
        keep = artists != seed_artist
        candidate_ids, distances = candidate_ids[keep], distances[keep]

    seed_genre = _genre_top(tracks, seed_id) if seed_id is not None else None

    rng = np.random.default_rng(seed)
    out = {}
    for label, (lo, hi) in [("Familiar", familiar), ("Bridge", bridge), ("Adventurous", adventurous)]:
        hi = min(hi, len(candidate_ids))
        pool_idx = np.arange(lo, hi)

        allowed = gate_factors.get(label) if label in gate_factors else "skip"
        allowed_set = set(allowed or ALL_FACTORS) if allowed != "skip" else None
        if allowed_set is not None and seed_id is not None and len(pool_idx):
            connected = [i for i in pool_idx if set(cf.factors_between(seed_id, candidate_ids[i])) & allowed_set]
            pool_for_pick = connected if len(connected) >= n else list(pool_idx)
        else:
            pool_for_pick = list(pool_idx)

        share = max_same_genre_share.get(label) if label in max_same_genre_share else None
        if share is not None:
            pick = _cap_same_genre(tracks, cf, pool_for_pick, candidate_ids, seed_id, seed_genre, n, share,
                                    allowed_set, rng, widen_to=widen_to.get(label))
        else:
            pick = list(rng.choice(pool_for_pick, size=min(n, len(pool_for_pick)), replace=False)) if pool_for_pick else []
        pick = sorted(pick)
        factors = [cf.factors_between(seed_id, candidate_ids[i]) if seed_id is not None else [] for i in pick]
        out[label] = [(candidate_ids[i], distances[i], i + 1, f) for i, f in zip(pick, factors)]
    return out


def tier_for_adventure(adventure: float) -> str:
    """Map the frontend's single 0..1 slider onto one of the three tiers.

    Even split for now -- notebook 09 tuned the bands themselves, not where
    a continuous dial should switch between them.
    """
    if not 0.0 <= adventure <= 1.0:
        raise ValueError(f"adventure must be between 0 and 1, got {adventure}")
    if adventure < 1 / 3:
        return "Familiar"
    if adventure < 2 / 3:
        return "Bridge"
    return "Adventurous"
