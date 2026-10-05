---
title: Tasty (demo)
emoji: 🎧
colorFrom: pink
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Music discovery from familiar to adventurous
---

# Tasty (demo)

Pick a seed track and slide from **Familiar** through **Bridge** to **Adventurous**:
recommendations come from Discogs-EffNet audio embeddings over ~25k Free Music
Archive tracks, and each one says what connects it to the seed (genre, era,
geography). Search by name, genre, or by describing a sound ("dreamy synth pads").

Audio: [Free Music Archive](https://freemusicarchive.org/) (fma_medium), each
track under its artist's Creative Commons licence.

Built from the `deploy/` folder of the capstone repository
(`uv run python deploy/build_space.py`).
