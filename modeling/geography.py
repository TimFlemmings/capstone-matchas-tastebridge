"""Resolving `artist.location` free text down to a country, for the Bridge/
Explore "geography" connecting factor.

The field is self-reported and unstructured -- "Colchester UK", "San
Qunetin", "Gijón, Asturias", a handful of email addresses and in-jokes
("#somewhereOverTheMerzbow", "TERRA"). No geocoding API is used: only 754
distinct strings appear across all of fma_small, so a layered local resolver
(country aliases, US states, a short list of scene cities that show up
repeatedly) gets to 97.5% of the tracks that had *any* location string,
deterministically and with no network dependency. The remaining ~2.5% and the
32% of tracks with no location at all are left unresolved rather than
guessed -- callers should treat a missing country as "unknown", not "no
match", and skip the geography factor entirely for that track rather than
counting it against it.

    from modeling.geography import artist_countries
    countries = artist_countries()   # pd.Series, index=track_id, values=country name or NaN
"""

import re
from functools import lru_cache
from logging import getLogger
from pathlib import Path

import pandas as pd
import pycountry

logger = getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "artist_countries.csv"

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY",
}
US_ABBR = set(US_STATES.values())

# Aliases and abbreviations pycountry's own names don't cover, or that appear
# often enough here to be worth a manual entry.
COUNTRY_ALIASES = {
    "usa": "United States", "u.s.a": "United States", "us": "United States",
    "united states of america": "United States", "britain": "United Kingdom",
    "uk": "United Kingdom", "u k": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom", "northern ireland": "United Kingdom",
    "the netherlands": "Netherlands", "nederlands": "Netherlands", "holland": "Netherlands",
    "nl": "Netherlands", "south korea": "Korea, Republic of", "russia": "Russian Federation",
    "brasil": "Brazil", "czech republic": "Czechia", "czechoslovakia": "Czechia",
    "washington dc": "United States", "dc": "United States",
    "quebec": "Canada", "québec": "Canada", "manitoba": "Canada", "ontario": "Canada",
    "asturias": "Spain",
}

# Deliberately short: musicians self-report a scene city, not a country, and
# these are the ones that actually recur in fma_small -- not a full gazetteer.
CITY_COUNTRY = {
    "london": "United Kingdom", "manchester": "United Kingdom", "glasgow": "United Kingdom",
    "bristol": "United Kingdom", "leeds": "United Kingdom", "liverpool": "United Kingdom",
    "brighton": "United Kingdom", "coventry": "United Kingdom",
    "paris": "France", "berlin": "Germany", "hamburg": "Germany", "munich": "Germany",
    "cologne": "Germany", "leipzig": "Germany", "amsterdam": "Netherlands", "utrecht": "Netherlands",
    "nijmegen": "Netherlands", "rotterdam": "Netherlands", "brussels": "Belgium", "antwerp": "Belgium",
    "rome": "Italy", "milan": "Italy", "madrid": "Spain", "barcelona": "Spain",
    "gijon": "Spain", "gijón": "Spain", "lisbon": "Portugal", "vienna": "Austria",
    "zurich": "Switzerland", "geneva": "Switzerland", "lausanne": "Switzerland",
    "stockholm": "Sweden", "gothenburg": "Sweden", "oslo": "Norway", "copenhagen": "Denmark",
    "helsinki": "Finland", "warsaw": "Poland", "prague": "Czechia", "budapest": "Hungary",
    "athens": "Greece", "dublin": "Ireland", "galway": "Ireland",
    "moscow": "Russian Federation", "kyiv": "Ukraine", "kiev": "Ukraine",
    "istanbul": "Turkey", "ankara": "Turkey", "tel aviv": "Israel", "jerusalem": "Israel",
    "tokyo": "Japan", "osaka": "Japan", "kyoto": "Japan", "shinjuku": "Japan",
    "seoul": "Korea, Republic of", "beijing": "China", "shanghai": "China",
    "hong kong": "Hong Kong", "taipei": "Taiwan, Province of China", "singapore": "Singapore",
    "bangkok": "Thailand", "manila": "Philippines", "jakarta": "Indonesia",
    "mumbai": "India", "delhi": "India", "bangalore": "India",
    "sydney": "Australia", "melbourne": "Australia", "brisbane": "Australia", "perth": "Australia",
    "auckland": "New Zealand", "wellington": "New Zealand",
    "toronto": "Canada", "montreal": "Canada", "montréal": "Canada", "vancouver": "Canada",
    "ottawa": "Canada", "calgary": "Canada", "edmonton": "Canada", "winnipeg": "Canada",
    "rimouski": "Canada", "halifax": "Canada",
    "mexico city": "Mexico", "sao paulo": "Brazil", "são paulo": "Brazil",
    "rio de janeiro": "Brazil", "buenos aires": "Argentina", "lima": "Peru",
    "bogota": "Colombia", "santiago": "Chile", "cairo": "Egypt", "lagos": "Nigeria",
    "cape town": "South Africa", "johannesburg": "South Africa", "nairobi": "Kenya",
    "new york": "United States", "new york city": "United States", "nyc": "United States",
    "brooklyn": "United States", "manhattan": "United States", "queens": "United States",
    "harlem": "United States", "bushwick": "United States",
    "los angeles": "United States", "san francisco": "United States", "chicago": "United States",
    "boston": "United States", "seattle": "United States", "portland": "United States",
    "austin": "United States", "houston": "United States", "dallas": "United States",
    "atlanta": "United States", "miami": "United States", "philadelphia": "United States",
    "baltimore": "United States", "washington": "United States", "detroit": "United States",
    "minneapolis": "United States", "denver": "United States", "phoenix": "United States",
    "san diego": "United States", "nashville": "United States", "new orleans": "United States",
    "pittsburgh": "United States", "cleveland": "United States", "st. louis": "United States",
    "oakland": "United States", "milwaukee": "United States",
}


@lru_cache(maxsize=1)
def _country_names() -> dict:
    names = {c.name.lower(): c.name for c in pycountry.countries}
    for c in pycountry.countries:
        if hasattr(c, "common_name"):
            names[c.common_name.lower()] = c.name
    return names


def _clean(part: str) -> str:
    part = re.sub(r"\b\d{4,6}\b", "", part)  # drop zip-code-looking numbers
    part = part.replace(".", "")             # "U.K." / "D.C." -> "UK" / "DC"
    return re.sub(r"\s+", " ", part).strip()


def resolve_location(raw: str) -> str | None:
    """Best-effort country for one free-text `artist.location` value, or None.

    Tries, in order: a whole-string or comma/dash/slash-separated-part match
    against country names/aliases, US state names or 2-letter abbreviations,
    then a short list of scene cities. Ambiguous or unrecognisable text (a
    multi-city list, a joke, an email address, a bare ZIP code) returns None
    rather than a guess.
    """
    if not isinstance(raw, str):
        return None
    s = re.sub(r"[\r\n\t]+", " ", raw).strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None

    parts = [p.strip(" ()") for p in re.split(r"[,/\-]", s)]
    candidates = []
    for p in parts:
        p = _clean(p)
        if not p:
            continue
        candidates.append(p)
        words = p.split()
        if len(words) > 1:
            candidates.append(words[-1])
            candidates.append(" ".join(words[-2:]))

    names = _country_names()
    for cand in reversed(candidates):
        cl = cand.lower().strip(" ()")
        if cl in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[cl]
        if cl in names:
            return names[cl]
        if cl in US_STATES:
            return "United States"
        if cand.strip().upper() in US_ABBR and len(cand.strip()) == 2:
            return "United States"
        if cl in CITY_COUNTRY:
            return CITY_COUNTRY[cl]

    whole = _clean(s).lower().strip(" ()")
    if whole in CITY_COUNTRY:
        return CITY_COUNTRY[whole]
    if whole in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[whole]
    return None


def build_artist_countries(tracks: pd.DataFrame = None) -> pd.Series:
    """Resolved country per track_id. Only resolves each distinct location
    string once, then broadcasts -- 754 unique strings for 8,000 tracks."""
    if tracks is None:
        from modeling.data import load_tracks
        tracks = load_tracks(None)  # every subset, so this is reusable beyond fma_small

    loc = tracks[("artist", "location")]
    unique_locs = loc.dropna().unique()
    lookup = {u: resolve_location(u) for u in unique_locs}
    resolved = loc.map(lookup)
    resolved.name = "country"
    logger.info("Resolved %s / %s unique location strings; %s / %s tracks covered",
                sum(v is not None for v in lookup.values()), len(unique_locs),
                resolved.notna().sum(), len(tracks))
    return resolved


def load(path: Path = OUT_PATH) -> pd.Series:
    return pd.read_csv(path, index_col=0)["country"]


def _main():
    from modeling.data import load_tracks
    tracks = load_tracks(None)
    countries = build_artist_countries(tracks)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    countries.to_csv(OUT_PATH)
    print(f"wrote {countries.notna().sum()} / {len(countries)} resolved countries to {OUT_PATH}")
    print(countries.value_counts().head(15))


if __name__ == "__main__":
    _main()
