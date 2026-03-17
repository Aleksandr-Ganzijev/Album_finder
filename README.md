# 🎵 Album Finder

A music recommendation web app that suggests albums similar to one you love, using a combination of **Spotify audio features** and **Rate Your Music (RYM) genre tags**.

Built with **FastAPI**, **pandas**, and **scikit-learn**.

---

## 🚀 How It Works

Album Finder combines two sources of data to compute similarity between albums:

| Source          | Features                                                                                              | Weight |
| --------------- | ----------------------------------------------------------------------------------------------------- | ------ |
| Spotify         | danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo | 20%    |
| Rate Your Music | primary genres, secondary genres, descriptors                                                         | 80%    |

### 🎧 Audio Features

* Scaled using `StandardScaler` to normalize feature ranges
* L2-normalized so each album lies on a unit hypersphere
* Enables effective cosine similarity comparisons

### 🏷️ RYM Tags

* Processed using **TF-IDF weighting**
* Rare tags (e.g. "lowercase") carry more weight than common ones (e.g. "rock")
* Primary genres are duplicated to increase importance
* L2-normalized for compatibility with cosine similarity

### 🔗 Final Representation

* Audio + tag vectors are concatenated into a single sparse matrix
* **Cosine similarity** is computed against all other albums

---

## ✨ Features

* 🎯 **3-tier search**: exact match → starts-with → contains fallback
* ⭐ **Minimum score filter**: restrict results by Pitchfork rating
* 📊 **Dual ratings**: shows both Pitchfork and RYM scores
* 🌐 **Simple web UI** available at `/ui`
* ⚡ **REST API** for programmatic access

---

## 📁 Project Structure

```
Album_finder/
├── album_finder_app.py   # FastAPI app (API + UI)
├── output-data.csv       # Pitchfork + Spotify dataset
├── rym_clean1.csv        # RYM genres and descriptors
└── README.md
```

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Aleksandr-Ganzijev/Album_finder.git
cd Album_finder
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn pandas scikit-learn scipy
```

### 3. Run the app

```bash
uvicorn album_finder_app:app --reload
```

---

## 💻 Usage

### Web UI

Open your browser at:

```
http://127.0.0.1:8000/ui
```

Enter:

* Album name
* Minimum Pitchfork score
* Number of recommendations

Then click **Recommend**.

### API

```
GET /recommend?album={title}&n={count}&min_score={score}
```

| Parameter | Type  | Default | Description                           |
| --------- | ----- | ------- | ------------------------------------- |
| album     | str   | —       | Album title (partial match supported) |
| n         | int   | 10      | Number of recommendations             |
| min_score | float | 7.5     | Minimum Pitchfork score               |

#### Example Request

```
GET /recommend?album=Disintegration&n=5&min_score=8.0
```

#### Example Response

```json
{
  "searched": {
    "album": "Disintegration [Deluxe Edition]",
    "artist": "The Cure"
  },
  "recommendations": [
    {
      "artist": "Slowdive",
      "album": "Souvlaki",
      "pitchfork_score": 9.0,
      "rym_score": 4.1,
      "similarity": 0.94
    }
  ]
}
```

---

## 📊 Datasets

### `output-data.csv`

Albums reviewed by Pitchfork with Spotify audio features.

Key columns:

* artist
* album
* score
* genre
* releaseYear
* Spotify audio feature columns (9 total)

### `rym_clean1.csv`

Albums from Rate Your Music with community-generated tags.

Key columns:

* release_name
* artist_name
* primary_genres
* secondary_genres
* descriptors
* avg_rating

### 🔗 Data Merging

* Albums are matched using normalized keys (`album_key`, `artist_key`)
* Matching is based on lowercase + stripped strings
* Unmatched albums fall back to audio-only similarity

---

## ⚠️ Known Limitations

* Album matching relies on exact string normalization, so some albums may not merge correctly
* Albums without RYM matches only use Spotify audio features
* Recommendations are filtered by minimum score, but search scans the full dataset

---

## 📦 Dependencies

* fastapi
* pandas
* scikit-learn
* scipy
* uvicorn

---

## 📌 Future Improvements (Optional Ideas)

* Improve album matching with fuzzy string matching
* Add artist-based filtering
* Include release year weighting
* Enhance UI with sorting and filtering options

---

## 🧠 Summary

Album Finder blends **audio analysis** with **crowdsourced genre tagging** to deliver nuanced, high-quality album recommendations. By weighting semantic genre data more heavily than raw audio features, it captures both *sound* and *style* similarity.
