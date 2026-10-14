---
title: Tastey (demo)
emoji: 🎧
colorFrom: pink
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Music discovery from close to home to far out
---

# Tastey (demo)

Pick a seed track and turn the dial from **Home** through **Bridge** to **Adventure**:
15 picks from Discogs-EffNet audio embeddings over ~25k Free Music Archive tracks,
each showing its taste distance in bridgons and what connects it to the seed
(genre, era, region). Search by name, genre, or by describing a sound
("dreamy synth pads"). Your journey shows up on the Taste Map.

Audio: [Free Music Archive](https://freemusicarchive.org/) (fma_medium), each
track under its artist's Creative Commons licence.

Built from the `deploy/` folder of the capstone repository
(`uv run python deploy/build_space.py --push mrln77/tastey-demo`).
