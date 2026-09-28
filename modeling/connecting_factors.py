"""What makes a Bridge or Explore recommendation *explicable*, not just a
distance band.

A track within the right audio-distance range of a seed still needs an
interpretable reason a listener could point to. Four factors, each checked
independently and skipped (never counted as a mismatch) when either track is
missing the data it needs:

    genre        -- shares a `genres_all` tag directly
    genre_lineage -- different tag, but shares a top-level genre ancestor
                     (genres.csv's `top_level`; 100% coverage, unused until now)
    era          -- `album.date_released` within `ERA_YEARS` of each other
                     (67% coverage in fma_small)
    geography    -- same resolved country, via `modeling.geography`
                     (66.7% coverage in fma_small)

Two things measured and worth knowing before trusting these:

`genre_lineage` is currently inert on fma_small specifically -- checked all
8,000 tracks, and not one has `genres_all` spanning more than one top-level
genre. FMA-small was built by balancing exactly 8 top-level categories at
1,000 tracks each, so `genre_top` and top-level genre are the same partition
here; lineage beyond direct `genres_all` overlap has nothing left to find on
this subset. It's kept because it may do real work on `fma_medium` (16
top-level categories actually present, so a track can plausibly share
lineage without sharing a tag) -- just don't expect it to fire on fma_small.

`era`'s default (`ERA_YEARS = 2`) is already tight for a reason: this
catalogue is heavily clustered around 2009-2016 (std ~4.4 years), so even a
+-1 year window matches 27% of *random* pairs, and +-10 years matches 94%.
Widening this past ~2-3 years makes "era" nearly meaningless as a filter.

Tempo/rhythm was considered and left out: FOR_TIM.md's own measurement found
raw tempo carries almost no signal (eta^2 = 0.019 against genre, the weakest
of the rhythm block, undermined by octave-ambiguity in the estimate) -- not a
reliable basis for "these two connect because they're a similar tempo."

    from modeling.connecting_factors import ConnectingFactors
    cf = ConnectingFactors.build(tracks)
    cf.factors_between(seed_id, candidate_id)   # -> ["genre", "era"], or []
"""

import ast
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

import pandas as pd

logger = getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ERA_YEARS = 2  # tight on purpose -- see module docstring; this catalogue is heavily
               # clustered in time, so a loose window matches almost everything


@dataclass
class ConnectingFactors:
    genres_all: pd.Series          # track_id -> set of genre ids
    top_level: dict                # genre id -> top_level genre id
    year: pd.Series                # track_id -> release year or NaN
    country: pd.Series             # track_id -> resolved country or NaN
    era_years: int = ERA_YEARS

    @classmethod
    def build(cls, tracks: pd.DataFrame, genres_csv: Path = None, countries: pd.Series = None) -> "ConnectingFactors":
        genres_csv = genres_csv or REPO_ROOT / "data" / "raw" / "fma_metadata" / "genres.csv"
        genres_meta = pd.read_csv(genres_csv, index_col=0)
        top_level = genres_meta["top_level"].to_dict()

        genres_all = tracks[("track", "genres_all")].apply(lambda s: set(ast.literal_eval(s)))
        year = pd.to_datetime(tracks[("album", "date_released")], errors="coerce").dt.year

        if countries is None:
            from modeling.geography import load as load_countries
            try:
                countries = load_countries()
            except FileNotFoundError:
                logger.warning("data/artist_countries.csv missing -- run `python -m modeling.geography` "
                               "first; geography will be skipped as a connecting factor")
                countries = pd.Series(dtype=object)
        countries = countries.reindex(tracks.index)

        return cls(genres_all=genres_all, top_level=top_level, year=year, country=countries)

    def _top_levels(self, track_id) -> set:
        gids = self.genres_all.get(track_id, set())
        return {self.top_level[g] for g in gids if g in self.top_level}

    def factors_between(self, a, b) -> list:
        """Every factor connecting tracks a and b. Order is presentation order,
        roughly strongest/most legible first."""
        factors = []

        genres_a, genres_b = self.genres_all.get(a, set()), self.genres_all.get(b, set())
        if genres_a & genres_b:
            factors.append("genre")
        elif self._top_levels(a) & self._top_levels(b):
            factors.append("genre_lineage")

        ya, yb = self.year.get(a), self.year.get(b)
        if pd.notna(ya) and pd.notna(yb) and abs(ya - yb) <= self.era_years:
            factors.append("era")

        ca, cb = self.country.get(a), self.country.get(b)
        if pd.notna(ca) and pd.notna(cb) and ca == cb:
            factors.append("geography")

        return factors

    def describe(self, track_id) -> str:
        """A short human-readable tag for one track, for display alongside a
        recommendation -- not used for matching, just for showing your work."""
        bits = []
        y = self.year.get(track_id)
        if pd.notna(y):
            bits.append(str(int(y)))
        c = self.country.get(track_id)
        if pd.notna(c):
            bits.append(c)
        return ", ".join(bits) if bits else "(no era/geography data)"
