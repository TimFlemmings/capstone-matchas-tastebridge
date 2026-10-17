<p align="center"><img src="brand/tastey-vinyl-colour.png" width="96" alt="Tastey logo: a vinyl record drawn as a planet"></p>

# TasteBridge · the Tastey app

**Music discovery where you choose how far to travel from your taste.** Pick one seed track, then turn a dial from **Home** (sounds like it) through **Bridge** (related, but new) to **Adventure** (somewhere else entirely). Every pick shows how far it is from where you started, and why it connects.

**Try it:** https://mrln77-tastey-demo.hf.space · no sign-in, no listening history needed.

TasteBridge is the project; **Tastey** is the app. It's a neue fische Data Science capstone built on ~25,000 open-licence tracks from the [Free Music Archive](https://github.com/mdeff/fma).

---

## How it works

1. **A fingerprint per track.** Each 30-second clip is turned into 1,280 numbers by [Discogs-EffNet](https://essentia.upf.edu/models.html), a model pre-trained to recognise musical styles. Tracks that sound alike have similar fingerprints.
2. **One ranking per seed.** For a seed track, the whole catalogue is ranked by cosine distance to its fingerprint.
3. **Three slices of that ranking.** The dial picks 15 tracks from one band of ranks:

    | Dial | Ranks | Extra rules |
    |---|---|---|
    | Home | 1–40 | the nearest tracks, seed's own artist excluded |
    | Bridge | 81–420 | prefers the same era; caps on the seed's genre so other genres get a turn |
    | Adventure | 1,401–6,000 | must share a genre, era or region link but come from another decade; prefers other countries |

    The gaps between bands keep the levels clearly different. In Bridge and Adventure, other genres take turns, each contributing its closest track, so a small genre (Jazz for a Hip-Hop seed, say) still gets a slot.
4. **Taste distance in bridgons.** Each pick shows its distance from the seed as cosine distance × 100: under 30 stays close to home, 40 and up takes you somewhere new.
5. **Describe a sound.** Search also understands descriptions like "dreamy synth pads", via [CLAP](https://huggingface.co/laion/larger_clap_general) text–audio embeddings.

The rules live in [`modeling/tiers.py`](modeling/tiers.py); the bands were tuned by ear in notebook 09.

## Results

We judged similarity **by ear**: raters heard two clips blind (no names, no genres) and scored how alike they sounded from 1 to 5, with random pairs mixed in as a control.

| Finding | Number |
|---|---|
| Baseline (FMA's 518 features) vs random, round 1 | 2.96 vs 1.87 / 5 (p = 0.002) |
| Agreement with listeners: EffNet vs our own `heard` features | ρ = 0.73 vs 0.33 |
| Live dial, round 4 (pre-registered; 6 seeds, 42 pairs, 6 raters) | Home 3.60 · Bridge 3.35 · Adventure 1.71 · random 1.72 / 5 |
| Do listeners hear the dial's order? | yes: ρ = −0.63, p < 0.0001 |

Home and Bridge beat random; Adventure doesn't yet. It travels too far and loses its link to the seed, which is the next thing to fix.

Full write-ups with figures: [round 4, the dial](reports/round4/RESULTS.md) · [why the app moved from `heard` to EffNet](reports/effnet_vs_heard/README.md). Every round's design, pre-registration and anonymised ratings are in [`data/listening_test/`](data/listening_test/).

## The app

- **Start:** a first-visit picker with search, genre chips, starting points, Surprise me or Skip.
- **Discover:** the playlist, a pinned seed panel with its waveform, and the dial. *Seed this track* moves the journey on.
- **Library:** saved tracks with genre, location, year and BPM; sort manually or by any of them.
- **Taste Map:** your journey as a Tree or a Map, each step labelled with why it connects (same genre, era or region, and the distance).

Your journey and Library are stored only in your own browser; the server keeps nothing per person.

## Repository map

| Path | What's there |
|---|---|
| [`app_effnet.py`](app_effnet.py) | The app's FastAPI server (what the live site runs) |
| [`modeling/`](modeling/) | Recommender: `tiers.py` (the dial), `catalogue.py` (25k-track catalogue), `effnet_recommender.py`, `connecting_factors.py`, `text_search.py`, plus the feature and embedding pipelines |
| [`frontend/`](frontend/) | React + Vite app |
| [`notebooks/`](notebooks/) | The analysis, in order (below) |
| [`deploy/`](deploy/) | Builds and pushes the Hugging Face Space |
| [`brand/`](brand/) | Logo files |
| [`data/processed/`](data/processed/) | Small derived files the app and notebooks need; everything large stays out of Git |
| [`data/listening_test/`](data/listening_test/) | The listening tests: pair keys, pre-registrations (rounds 2–4) and every round's ratings, raters as anonymous codes |
| [`reports/`](reports/) | Written-up results with figures (round 4, EffNet vs `heard`) |
| `modeling/round2.py` … `round4_analysis.py`, `embedding_vs_ear.py`, `effnet_vs_heard.py` | Build each listening-test round and score models against the ratings |
| [`api/main.py`](api/main.py), [`API.md`](API.md) | A lighter API with the same recommender, documented with examples |
| [`app.py`](app.py) | The earlier 8,000-track PCA prototype |
| [`docs/`](docs/), [`HANDOFF.md`](HANDOFF.md) | Working notes and the project's history so far |

**Notebooks:** 01 EDA (and `01_eda_baseline`) · 01A random recommender · 02–04 baselines (518 features, PCA-146, MFCC) · 02/03 listening test and results · 04 new features (`heard`) · 05 multi-seed taste · 05a tiered evaluation · 06 `heard` check · 07 tiers and EchoNest cross-check · 08 Discogs-EffNet · 09 dial tuning · 10 taste map · 11 the 25k catalogue's 45 checks.

## Run it locally

See **[SETUP.md](SETUP.md)**. In short: `uv sync`, fetch the embeddings from the project's Hugging Face dataset, start `app_effnet.py`, then the front end.

## Deploy

```bash
uv run python deploy/build_space.py --push mrln77/tastey-demo
```

It rebuilds the bundle from your current code and uploads it to the [Space](https://huggingface.co/spaces/mrln77/tastey-demo), which rebuilds in 2–3 minutes. Embeddings and audio stay in the private dataset `mrln77/tastebridge-audio`, read with the Space's `HF_TOKEN` secret; tracks without a licence on record are left out.

## Data and licences

Music and metadata: the [Free Music Archive](https://freemusicarchive.org/) dataset ([FMA](https://github.com/mdeff/fma), fma_medium), each track under its artist's Creative Commons licence. Listening-test ratings are stored with raters as anonymous codes (R1, R2b, …), no names or emails.

## Team

- **Tim** ([@TimFlemmings](https://github.com/TimFlemmings))
- **Amin** ([@amintorabi7](https://github.com/amintorabi7))

## Limitations

Small listening-test samples (6 raters in round 4); Adventure not yet better than random; the Taste Map not yet tested by ear; mood poorly captured; open music only, no mainstream catalogue.
