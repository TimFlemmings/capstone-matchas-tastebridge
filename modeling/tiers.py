"""The tuned Familiar / Bridge / Adventurous tier system, as code.

Extracted from notebook 09 (round 3's settings) so the API and notebook 10
share one implementation instead of two copies drifting apart. Needs only
`EffNetRecommender` and `ConnectingFactors` — no Essentia, no audio I/O, so it
runs in the project's normal (3.13) environment.

    from modeling.tiers import build_connecting_factors, explore_tiers
    cf = build_connecting_factors(rec, tracks)
    explore_tiers(rec, cf, tracks, seed_id=2, seed_artist="AWOL")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.connecting_factors import ConnectingFactors
from modeling.effnet_recommender import EffNetRecommender

ALL_FACTORS = ("genre", "genre_lineage", "related_genre", "era", "geography")

# Round 3's tuned defaults from notebook 09 -- the settings that "felt right"
# on the Hip-Hop / Instrumental / Pop catalogue seeds plus the external seed.
DEFAULT_BANDS = {"familiar": (0, 15), "bridge": (30, 150), "adventurous": (500, 2000)}
DEFAULT_GATE_FACTORS = {"Bridge": ("era",), "Adventurous": None}
DEFAULT_MAX_SAME_GENRE_SHARE = {"Bridge": 0.25, "Adventurous": 0.2}
DEFAULT_MAX_SISTER_GENRE_SHARE = {"Bridge": 0.4, "Adventurous": 0.2}
DEFAULT_WIDEN_TO = {"Bridge": DEFAULT_BANDS["adventurous"][0], "Adventurous": 3000}
DEFAULT_EXCLUDE_SAME_DECADE = {"Adventurous"}

# The same bands for the 25k fma_medium catalogue (`modeling.catalogue`). The
# defaults above were tuned by ear on fma_small's 7,994 tracks; in a catalogue
# 3x the size, a given rank sits ~3x closer to the seed, so the old numbers
# would make Bridge and Adventurous audibly tamer. Each boundary is moved to the
# rank where the *same cosine distance* now falls (median over 300 fma_small
# seeds): 15->40, 30->80, 150->420, 500->1400, 2000->6000, 3000->9000.
MEDIUM_BANDS = {"familiar": (0, 40), "bridge": (80, 420), "adventurous": (1400, 6000)}
MEDIUM_WIDEN_TO = {"Bridge": MEDIUM_BANDS["adventurous"][0], "Adventurous": 9000}
DEFAULT_PREFER_DIFFERENT_COUNTRY = {"Adventurous"}

# How many picks the app serves per request. Fixed server-side (the request
# carries no `limit`): the genre-share caps are fractions of this, so at 15
# Bridge keeps 6 slots and Adventurous 9 for genres other than the seed's own
# and its sister genres.
RECOMMENDATION_COUNT = 15

# How the distant-genre slots are filled (see _cap_genre_diversity):
#   "rank"       -- the n closest distant-genre tracks, whatever their genre
#                   (rounds 1-3). The nearest big genres take every slot.
#   "nearest"    -- genres take turns, each contributing its closest track;
#                   genres ordered by how close their closest track is.
#   "enrichment" -- same turns, but genres ordered by how over-represented
#                   they are around the seed vs. their catalogue share.
DISTANT_ORDERS = ("rank", "nearest", "enrichment")
DEFAULT_DISTANT_ORDER = "enrichment"


def genre_sister_map(rec: EffNetRecommender, tracks: pd.DataFrame, k: int = 2) -> dict:
    """genre_top -> its k nearest *other* genre_top by mean-embedding cosine
    distance -- e.g. Instrumental's nearest is Experimental (dist 0.080), a
    relationship FMA's own genre tree can't express here (genres.csv's
    top_level never crosses genre_top on fma_small, so `genre_lineage` alone
    can't connect Instrumental to Experimental -- see connecting_factors.py).
    This is grounded in the same embedding space driving the recommendations,
    not a hand-curated guess."""
    emb = rec.embedding_
    genre_top = tracks.loc[emb.index, ("track", "genre_top")]
    centroids = {}
    for g in genre_top.dropna().unique():
        v = emb.loc[genre_top[genre_top == g].index].to_numpy()
        c = v.mean(axis=0)
        centroids[g] = c / np.linalg.norm(c)
    genres = list(centroids.keys())
    sisters = {}
    for g in genres:
        dists = {g2: 1 - centroids[g] @ centroids[g2] for g2 in genres if g2 != g}
        sisters[g] = set(sorted(dists, key=dists.get)[:k])
    return sisters


def build_connecting_factors(rec: EffNetRecommender, tracks: pd.DataFrame, k: int = 2,
                             genres_csv=None) -> ConnectingFactors:
    """ConnectingFactors with `related_genre` wired to this embedding's sister map."""
    return ConnectingFactors.build(tracks, genres_csv=genres_csv,
                                   genre_centroids=genre_sister_map(rec, tracks, k=k))


def _genre_top(tracks: pd.DataFrame, track_id):
    return tracks.loc[track_id, ("track", "genre_top")]


def _different_decade(cf: ConnectingFactors, seed_id, candidate_id):
    """True unless both years are known AND fall in the same decade -- an
    unknown year can't be proven same-decade, so it passes rather than being
    penalized (same policy as every other factor here: missing data is
    skipped, never counted as a mismatch)."""
    ys, yc = cf.year.get(seed_id), cf.year.get(candidate_id)
    if pd.isna(ys) or pd.isna(yc):
        return True
    return int(ys) // 10 != int(yc) // 10


def _cap_genre_diversity(tracks, cf, pool_idx, candidate_ids, seed_id, seed_genre, n,
                          max_same_share, max_sister_share, allowed_set, rng,
                          widen_to=None, prefer_different_country=False, seed_country=None,
                          distant_order="rank", require_different_decade=False):
    """Split pool_idx into same-genre / sister-genre (cf.genre_centroids) /
    distant-genre buckets relative to seed_genre. Same- and sister-genre
    picks only ever come from pool_idx itself, each capped at
    floor(n * its max_share) -- None means uncapped. Distant-genre picks come
    from pool_idx first; if that's not enough, search further out in rank
    (up to widen_to, exclusive) for more, applying the same allowed_set
    connection gate to anything found there. Any shortfall backfills from
    sister first, then same, as a last resort -- sister is still "connected",
    same is the least interesting fallback.

    When prefer_different_country is set, within the sister/distant buckets
    candidates whose resolved country matches seed_country are pushed behind
    ones that differ or are unknown -- a soft preference (reordering, not a
    cut), since geography coverage is only ~67% and a hard requirement would
    starve tightly-clustered seeds.

    distant_order picks how the distant slots are shared out -- see
    DISTANT_ORDERS. With "nearest"/"enrichment" every distant genre found in
    the pool *or* out to widen_to is a candidate, so a small genre whose
    closest track sits past the band still gets its turn. Anything found
    past the band passes the same connection gate and, with
    require_different_decade, the same decade filter as the pool did."""
    if distant_order not in DISTANT_ORDERS:
        raise ValueError(f"distant_order must be one of {DISTANT_ORDERS}, got {distant_order!r}")
    pool_idx = list(pool_idx)
    if seed_genre is None or not pool_idx:
        # no genre to compare against (the external seed has none) -- fall
        # back to a plain n-sized sample, NOT the whole unfiltered band
        return list(rng.choice(pool_idx, size=min(n, len(pool_idx)), replace=False)) if pool_idx else []

    sister_genres = cf.genre_centroids.get(seed_genre, set())

    def bucket(i):
        g = _genre_top(tracks, candidate_ids[i])
        if g == seed_genre:
            return "same"
        return "sister" if g in sister_genres else "distant"

    def deprioritize_same_country(idxs):
        if not prefer_different_country or seed_country is None or pd.isna(seed_country):
            return idxs
        different, same_country = [], []
        for i in idxs:
            c = cf.country.get(candidate_ids[i])
            (same_country if c == seed_country else different).append(i)
        return different + same_country

    same_idx = [i for i in pool_idx if bucket(i) == "same"]
    sister_idx = [i for i in pool_idx if bucket(i) == "sister"]
    distant_idx = [i for i in pool_idx if bucket(i) == "distant"]

    max_same = int(np.floor(n * max_same_share)) if max_same_share is not None else n
    max_sister = int(np.floor(n * max_sister_share)) if max_sister_share is not None else n
    n_same = min(max_same, len(same_idx))
    n_sister = min(max_sister, len(sister_idx))
    needed_distant = n - n_same - n_sister

    end = min(widen_to, len(candidate_ids)) if widen_to is not None else pool_idx[-1] + 1
    genres = cf.genre_top.reindex(candidate_ids[:max(end, pool_idx[-1] + 1)]).to_numpy()
    near_genres = {seed_genre} | sister_genres

    def eligible_beyond_band(j):
        cid = candidate_ids[j]
        if require_different_decade and seed_id is not None and not _different_decade(cf, seed_id, cid):
            return False
        if allowed_set is not None and seed_id is not None:
            return bool(set(cf.factors_between(seed_id, cid)) & allowed_set)
        return True

    if distant_order == "rank":
        for j in range(pool_idx[-1] + 1, end):
            if len(distant_idx) >= needed_distant:
                break
            if genres[j] not in near_genres and eligible_beyond_band(j):
                distant_idx.append(j)  # appended in rank order (closest-first), since j increases
        distant_pick = deprioritize_same_country(distant_idx)[:needed_distant]
    else:
        by_genre = {}  # distant genre -> its eligible candidates, closest-first
        for i in distant_idx:
            by_genre.setdefault(genres[i] if pd.notna(genres[i]) else None, []).append(i)
        for j in range(pool_idx[-1] + 1, end):
            g = genres[j] if pd.notna(genres[j]) else None
            if g in near_genres or len(by_genre.get(g, ())) >= needed_distant:
                continue
            if eligible_beyond_band(j):
                by_genre.setdefault(g, []).append(j)
        by_genre.pop(None, None)  # no genre_top: nothing to take turns as

        if distant_order == "nearest":
            order = sorted(by_genre, key=lambda g: by_genre[g][0])
        else:
            local = pd.Series(genres[:end]).value_counts(normalize=True)
            overall = cf.genre_top.value_counts(normalize=True)
            order = sorted(by_genre, key=lambda g: (-local.get(g, 0.0) / overall[g], by_genre[g][0]))

        queues = [deprioritize_same_country(by_genre[g]) for g in order]
        distant_pick, turn = [], 0
        while len(distant_pick) < needed_distant and any(len(q) > turn for q in queues):
            distant_pick += [q[turn] for q in queues if len(q) > turn][:needed_distant - len(distant_pick)]
            turn += 1

    n_distant = len(distant_pick)
    shortfall = n - (n_same + n_sister + n_distant)
    if shortfall > 0:  # backfill: sister first (still "connected"), then same, as a last resort
        extra = min(shortfall, len(sister_idx) - n_sister)
        n_sister += extra
        shortfall -= extra
    if shortfall > 0:
        n_same = min(len(same_idx), n_same + shortfall)

    same_pick = list(rng.choice(same_idx, size=n_same, replace=False)) if n_same else []
    sister_pick = deprioritize_same_country(sister_idx)[:n_sister]
    chosen = same_pick + sister_pick + distant_pick
    return chosen if chosen else pool_idx[: min(n, len(pool_idx))]


def explore_tiers(rec: EffNetRecommender, cf: ConnectingFactors, tracks: pd.DataFrame,
                   seed_id=None, seed_vec=None, seed_artist=None,
                   familiar=DEFAULT_BANDS["familiar"], bridge=DEFAULT_BANDS["bridge"],
                   adventurous=DEFAULT_BANDS["adventurous"], n=5,
                   gate_factors=None, max_same_genre_share=None, max_sister_genre_share=None,
                   widen_to=None, exclude_same_decade=None, prefer_different_country=None,
                   distant_order=None, only=None, seed=0):
    """Familiar/Bridge/Adventurous picks for one seed.

    Returns {label: [(track_id, distance, rank, factors), ...]}. Pass either
    `seed_id` (a catalogue track) or `seed_vec` (an already-embedded external
    track, unit-normalised) -- see notebook 09 for how the external case is
    built. `only` limits the work to one tier label (the API only ever
    serves one at a time). `cf` should come from `build_connecting_factors`
    so the sister-genre buckets and `related_genre` factor are populated.
    """
    gate_factors = DEFAULT_GATE_FACTORS if gate_factors is None else gate_factors
    # 0.0 = never the seed's own genre_top; 0.25 = up to ~1/4 of picks may;
    # None = no cap. max_sister_genre_share works the same way but against
    # cf.genre_centroids' sister set (e.g. Experimental for an Instrumental
    # seed) -- capping it pushes picks toward genuinely distant genres
    # instead of the audibly-closest-but-still-obvious neighbour.
    max_same_genre_share = DEFAULT_MAX_SAME_GENRE_SHARE if max_same_genre_share is None else max_same_genre_share
    max_sister_genre_share = DEFAULT_MAX_SISTER_GENRE_SHARE if max_sister_genre_share is None else max_sister_genre_share
    # how far out (exclusive upper rank bound) a tier may search for
    # distant-genre candidates once its own band runs short
    widen_to = DEFAULT_WIDEN_TO if widen_to is None else widen_to
    # tier labels for which the candidate pool is pre-filtered to a different
    # decade than the seed (unknown years pass through -- see _different_decade)
    exclude_same_decade = DEFAULT_EXCLUDE_SAME_DECADE if exclude_same_decade is None else set(exclude_same_decade)
    # tier labels for which same-country candidates are pushed to the back of
    # the pick order rather than excluded (soft preference, ~67% coverage)
    prefer_different_country = (DEFAULT_PREFER_DIFFERENT_COUNTRY if prefer_different_country is None
                                else set(prefer_different_country))
    distant_order = DEFAULT_DISTANT_ORDER if distant_order is None else distant_order

    if seed_vec is None:
        row = rec._position(seed_id)
        seed_vec = rec.embedding_.iloc[[row]]
    widen_values = [v for v in widen_to.values() if v is not None]
    search_n = max(adventurous[1] + 1, (max(widen_values) + 1) if widen_values else 0, 50)
    search_n = min(search_n, len(rec.track_ids_))
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
    seed_country = cf.country.get(seed_id) if seed_id is not None else None

    rng = np.random.default_rng(seed)
    out = {}
    for label, (lo, hi) in [("Familiar", familiar), ("Bridge", bridge), ("Adventurous", adventurous)]:
        if only is not None and label != only:
            continue
        hi = min(hi, len(candidate_ids))
        pool_idx = np.arange(lo, hi)

        if label in exclude_same_decade and seed_id is not None and len(pool_idx):
            pool_idx = np.array([i for i in pool_idx if _different_decade(cf, seed_id, candidate_ids[i])])

        allowed = gate_factors.get(label) if label in gate_factors else "skip"
        allowed_set = set(allowed or ALL_FACTORS) if allowed != "skip" else None
        if allowed_set is not None and seed_id is not None and len(pool_idx):
            connected = [i for i in pool_idx if set(cf.factors_between(seed_id, candidate_ids[i])) & allowed_set]
            pool_for_pick = connected if len(connected) >= n else list(pool_idx)
        else:
            pool_for_pick = list(pool_idx)

        same_share = max_same_genre_share.get(label) if label in max_same_genre_share else None
        sister_share = max_sister_genre_share.get(label) if label in max_sister_genre_share else None
        if same_share is not None or sister_share is not None:
            pick = _cap_genre_diversity(tracks, cf, pool_for_pick, candidate_ids, seed_id, seed_genre, n,
                                         same_share, sister_share, allowed_set, rng,
                                         widen_to=widen_to.get(label),
                                         prefer_different_country=label in prefer_different_country,
                                         seed_country=seed_country, distant_order=distant_order,
                                         require_different_decade=label in exclude_same_decade)
        else:
            pick = list(rng.choice(pool_for_pick, size=min(n, len(pool_for_pick)), replace=False)) if pool_for_pick else []
        pick = sorted(pick)
        factors = [cf.factors_between(seed_id, candidate_ids[i]) if seed_id is not None else [] for i in pick]
        out[label] = [(candidate_ids[i], distances[i], i + 1, f) for i, f in zip(pick, factors)]
    return out


def tier_for_adventure(adventure: float) -> str:
    """Map the frontend's 0..1 slider onto one of the three tiers.

    The slider only stops at 0 / 0.5 / 1 (Familiar / Bridge / Adventurous);
    the even split just keeps other API callers' in-between values sensible.
    """
    if not 0.0 <= adventure <= 1.0:
        raise ValueError(f"adventure must be between 0 and 1, got {adventure}")
    if adventure < 1 / 3:
        return "Familiar"
    if adventure < 2 / 3:
        return "Bridge"
    return "Adventurous"
