from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import math

# Load existing dataset with spotify and pitchfork
df = pd.read_csv("output-data.csv")
# Describe the spotify columns
feature_cols = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "tempo"]
# Drop the columns with missing values here
df = df.dropna(subset=["album", "artist", "score"] + feature_cols)

# Load RYM tags
rym = pd.read_csv("rym_clean1.csv")
# Normalize keys for a simple merge (improve matching later), creates new columns which should be easier to match
df["album_key"] = df["album"].str.lower().str.strip()
df["artist_key"] = df["artist"].str.lower().str.strip()
rym["album_key"] = rym["release_name"].str.lower().str.strip()
rym["artist_key"] = rym["artist_name"].str.lower().str.strip()

# Merge dataframes
merged = df.merge(
    rym[["album_key", "artist_key", "primary_genres", "secondary_genres", "descriptors", "avg_rating"]],
    on=["album_key", "artist_key"],
    how="left"
)

# This builds a weighted bag-of-tags for each album; repeat primary genres to weight them more
# "rock, rock, indie, shoegaze, psychedelic"
tag_text = (
    merged["primary_genres"].fillna("") + ", " +
    merged["primary_genres"].fillna("") + ", " +
    merged["secondary_genres"].fillna("") + ", " +
    merged["descriptors"].fillna("")
)

# This function splits the genres by commas
def comma_tokenizer(s: str):
    return [t.strip().lower() for t in s.split(",") if t.strip()]
# Use own tokenizer, don't touch text before tokenization, we already lowercase in function, do not use regex tokenization
# Each album becomes a vector (post-rock=0.42), rare tags = higher weight, common tags = lower wight,
# albums with similar tag distributions = higher similarity
vectorizer = TfidfVectorizer(
    tokenizer=comma_tokenizer,
    preprocessor=None,
    lowercase=False,
    token_pattern=None
)

# Here in practice tags are more accurate descriptors than spotify data, so I value it more
audio_weight = 0.2
tag_weight = 0.8

# Takes only the spotify audio columns and converts them into numpy array
numpy_spotify_audio = merged[feature_cols].to_numpy()

# Scaler.fit_transform makes it so everything become comparable, without it loudness and tempo overpower everything
# ^ Make all audio features speak the same numerical language
# normalize - each row/album vector becomes a point on a unit hypersphere for cos-similarity
scaler = StandardScaler()
numpy_spotify_audio_norm = normalize(scaler.fit_transform(numpy_spotify_audio))

# For every unique tag across albums create one column, value = TF-IDF weight
# A sciPy sparse matric is created
numpy_spotify_rym = vectorizer.fit_transform(tag_text)

# normalize - each album's tag vector becomes a point on a unit hypersphere for cos-similarity;
# should compare distribution not quantity of tags
numpy_spotify_rym_norm = normalize(numpy_spotify_rym)

# csr_matrix = Put audio in the same container type as tags (convert dense audio matrix to sparse to match TF-IDF format)
# ^ numpy to SciPy
# weigh with the values we chose
# hstack concatenates columns, not rows
numpy_spotify_combined = hstack([audio_weight * csr_matrix(numpy_spotify_audio_norm), tag_weight * numpy_spotify_rym_norm])


# CSR MATRIX!!!!!!
# [0, 0, 0, 2, 0, 5]
# data = [2, 5]
# indices = [3, 5]
# indptr = [0, 2]
# All main, secondary genres are included, as well as descriptors


# Start a fastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"home is where we are not present"}

def recommend_similar(album_title, data, feature_cols, n=10):
    query = album_title.lower().strip()
    albums_lower = data["album"].str.lower()

    # 1. Exact match
    exact = data[albums_lower == query]
    if not exact.empty:
        matches = exact
    else:
        # 2. Starts-with match
        starts = data[albums_lower.str.startswith(query, na=False)]
        if not starts.empty:
            matches = starts
        else:
            # 3. Fallback: contains
            contains = data[albums_lower.str.contains(query, na=False)]
            matches = contains

    if matches.empty:
        return [{"error": "Album not found"}]

    idx = matches.index[0]
    row_pos = data.index.get_loc(idx)
    subset_numpy_spotify_combined = numpy_spotify_combined[data.index, :]
    sims = cosine_similarity(subset_numpy_spotify_combined[row_pos:row_pos + 1], subset_numpy_spotify_combined).flatten()
    sim_df = data.copy()
    sim_df["similarity"] = sims
    def safe_float(x):
        return None if pd.isna(x) or (isinstance(x, float) and math.isnan(x)) else x
    results = (
        sim_df.sort_values("similarity", ascending=False)
        .iloc[1:n+1]
        .apply(
            lambda row: {
                "artist": row["artist"],
                "album": row["album"],
                "pitchfork_score": safe_float(row["score"]),
                "rym_score": safe_float(row.get("avg_rating", None)),
                "similarity": safe_float(row["similarity"]),
            },
            axis=1
        )
        .tolist()
    )
    return results

@app.get("/recommend")
def recommend(album: str, n: int = 10, min_score: float = 7.5):
    #1. Find the album in the full dataset
    mask = merged["album"].str.lower().str.contains(album.lower(), na=False)
    if mask.any():
        searched_album = df[mask].iloc[0]
        heading = {
            "album": searched_album["album"],
            "artist": searched_album["artist"],
        }
    else:
        return {"searched": {"error": "Album not found"}, "recommendations": []}
    #2. FIlter dataset fo recommendations only
    filtered_df=merged[merged["score"]>=min_score]
    #3. Generate recommendations from filtered_df
    recs = recommend_similar(searched_album["album"], filtered_df, feature_cols, n)
    return {"searched": heading, "recommendations": recs}

from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
<head>
    <title>Album Recommender</title>
</head>
<body>
    <h1>Album Recommender</h1>
    <input type="text" id="album" placeholder="Enter album name"/>
    <input type="number" id="min_score" step="0.1" placeholder="minimum album rating"/>
    <input type="number" id="n_elements" step="1" placeholder="number of recommended albums"/>
    <button onclick="getRecommendations()">Recommend</button>
    <h2 id="searched-album-heading"></h2>
    <ul id="results"></ul>

    <script>
        async function getRecommendations() {
            const album = document.getElementById('album').value;
            const heading = document.getElementById('searched-album-heading');

            const minScore = parseFloat(document.getElementById('min_score').value);
            const n_elements = parseInt(document.getElementById('n_elements').value);
            const res = await fetch(
                `/recommend?album=${encodeURIComponent(album)}&n=${n_elements}&min_score=${minScore}`
            );

            const data = await res.json();

            const results = document.getElementById('results');
            results.innerHTML = '';

            // Update heading
            if(data.searched && !data.searched.error) {
                heading.textContent = `Albums most similar to "${data.searched.album}" by ${data.searched.artist}`;
            } else {
                heading.textContent = `Album not found: "${album}"`;
            }

            // Display recommendations
            if (data.recommendations && data.recommendations.length > 0 && !data.recommendations[0].error) {
                data.recommendations.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = `${item.artist} - ${item.album} (Pitchfork: ${item.pitchfork_score}, RYM: ${item.rym_score}, Similarity: ${item.similarity.toFixed(2)})`;
                    results.appendChild(li);
                });
            } else {
                results.innerHTML = '<li>No recommendations found.</li>';
            }
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)