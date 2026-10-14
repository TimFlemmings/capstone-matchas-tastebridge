"""Find tracks by describing them: "dreamy synth pads", "punk rock with shouting".

Uses the CLAP embeddings from `modeling.clap_embed`. At query time only CLAP's
*text* half runs (125M parameters, ~50 ms a query on CPU), so a server never
loads the audio half; its output is identical to the full model's
(cosine 1.000). Scoring is one dot product against the precomputed track
vectors.

Notebook 11, part E: descriptive queries work well (all 14 tested reach >= 5x
lift); bare genre names that label a scene rather than a sound ("Experimental",
"Old-Time") don't. Suggest descriptions in the UI, not genre names.

    search = TextSearch(catalogue.ids, catalogue.clap)
    search.query("calm solo piano", n=10)   # -> [(track_id, score), ...]
"""

from __future__ import annotations

import numpy as np

from modeling.clap_embed import MODEL_NAME


class TextSearch:
    def __init__(self, ids: np.ndarray, clap: np.ndarray, model_name: str = MODEL_NAME):
        self.ids = np.asarray(ids)
        self.clap = clap / np.linalg.norm(clap, axis=1, keepdims=True)
        self.model_name = model_name
        self._model = None

    def load(self) -> "TextSearch":
        """Load the text model now rather than on the first query (a few seconds)."""
        if self._model is None:
            from transformers import AutoTokenizer, ClapTextModelWithProjection
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()  # otherwise it lists every skipped audio weight as "unexpected"
            self._model = (ClapTextModelWithProjection.from_pretrained(self.model_name).eval(),
                           AutoTokenizer.from_pretrained(self.model_name))
        return self

    def embed(self, text: str) -> np.ndarray:
        import torch

        model, tokenizer = self.load()._model
        with torch.no_grad():
            v = model(**tokenizer([text], return_tensors="pt", padding=True)).text_embeds[0].numpy()
        return v / np.linalg.norm(v)

    def query(self, text: str, n: int = 25) -> list[tuple[int, float]]:
        """The n tracks whose sound best matches `text`, best first, with cosine scores."""
        scores = self.clap @ self.embed(text)
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top]


def search_catalogue(tracks, search: TextSearch, query: str, n: int = 25) -> list[tuple[int, str]]:
    """The search box: [(track_id, matched_by), ...], at most n, best first.

    1. The query is exactly a genre ("jazz") -- that genre's tracks, ordered
       by how well they fit "<genre> music" (`matched_by="genre"`). CLAP on its
       own can't do this for genres that label a scene rather than a sound
       (Experimental, Old-Time: P@20 of 0 in notebook 11), but it can rank
       inside one. First, because otherwise "jazz" fills up with Old-Time
       tracks that merely have "jazz" in the title.
    2. Title or artist contains the query -- someone typing a name wants that
       name (`"name"`).
    3. Fill whatever is left with tracks that *sound* like the query (`"sound"`).
    """
    q = query.strip().casefold()
    if not q:
        return [(int(t), "name") for t in tracks.index[:n]]

    results, seen = [], set()

    def add(track_ids, matched_by):
        for t in track_ids:
            if len(results) >= n:
                return
            if int(t) not in seen:
                results.append((int(t), matched_by))
                seen.add(int(t))

    genre = tracks[("track", "genre_top")].fillna("").astype(str)
    genre_match = genre.str.casefold() == q
    if genre_match.any():
        in_genre = np.isin(search.ids, tracks.index[genre_match])
        scores = np.where(in_genre, search.clap @ search.embed(f"{genre[genre_match].iloc[0]} music"), -np.inf)
        order = np.argsort(-scores)[: int(in_genre.sum())]
        add(search.ids[order], "genre")

    title = tracks[("track", "title")].fillna("").astype(str).str.casefold()
    artist = tracks[("artist", "name")].fillna("").astype(str).str.casefold()
    add(tracks.index[title.str.contains(q, regex=False) | artist.str.contains(q, regex=False)], "name")

    if len(results) < n:
        add([t for t, _ in search.query(query, n=n + len(seen))], "sound")
    return results
