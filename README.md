# 🎵 Album Finder

A music recommendation web app that suggests albums similar to one you love,
based on a combination of **Spotify audio features** and **Rate Your Music genre tags**.

Built with **FastAPI**, **pandas**, and **scikit-learn**.

---

## How It Works

Album Finder combines two sources of data to compute similarity between albums:

| Source | Features | Weight |
|--------|----------|--------|
| Spotify | danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo | 20% |
| Rate Your Music | primary genres, secondary genres, descriptors | 80% |

**Audio features** are scaled with `StandardScaler` so that high-range values
like loudness and tempo don't overpower quieter features, then L2-normalized
so every album sits on a unit hypersphere for cosine similarity.

**RYM tags** are weighted with TF-IDF — rare tags like "lowercase" score higher
than common ones like "rock" — then also L2-normalized. Primary genres are
repeated twice in the tag string to give them extra influence.

Both vectors are concatenated into a single sparse matrix and **cosine similarity**
is computed against all other albums.

---

## Features

- 🎯 **3-tier search** — exact match → starts-with → contains fallback
- ⭐ **Minimum score filter** — only recommend albums above a set Pitchfork rating
- 📊 **Dual ratings** — results show both Pitchfork and RYM scores
- 🌐 **Simple web UI** — accessible at `/ui`, no setup needed
- ⚡ **REST API** — query programmatically via `/recommend`

---

## Project Structure

