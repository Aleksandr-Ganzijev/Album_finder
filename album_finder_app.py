from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# Load existing dataset with spotify and pitchfork
df = pd.read_csv("output-data.csv")
feature_cols = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "tempo"]
df = df.dropna(subset=["album", "artist", "score"] + feature_cols)
X = df[feature_cols].to_numpy()
app = FastAPI()
# Standadize audio features (important for cosine behavior)
scaler = StandardScaler()
#Load RYM tags and merge
rym = pd.read_csv("rym_clean1.csv")
# Normalize keys for a simple merge (improve matching later)
df["album_key"] = df["album"].str.lower().str.strip()
df["artist_key"] = df["artist"].str.lower().str.strip()
rym["album_key"] = rym["release_name"].str.lower().str.strip()
rym["artist_key"] = rym["artist_name"].str.lower().str.strip()
merged = df.merge(
    rym[["album_key", "artist_key", "primary_genres", "secondary_genres", "descriptors", "avg_rating"]],
    on=["album_key", "artist_key"],
    how="left"
)
# Build a single tag string; repeat primary genres to weight them more
tag_text = (
    merged["primary_genres"].fillna("") + ", " +
    merged["primary_genres"].fillna("") + ", " +
    merged["secondary_genres"].fillna("") + ", " +
    merged["descriptors"].fillna("")
)
def comma_tokenizer(s: str):
    return [t.strip().lower() for t in s.split(",") if t.strip()]
vectorizer = TfidfVectorizer(
    tokenizer=comma_tokenizer,
    preprocessor=None,
    lowercase=False,
    token_pattern=None
)
audio_weight = 0.2
tag_weight = 0.8
X_audio = merged[feature_cols].to_numpy()
X_audio_norm = normalize(scaler.fit_transform(X_audio))
X_rym = vectorizer.fit_transform(tag_text)
X_rym_norm = normalize(X_rym)

X_combined = hstack([audio_weight * csr_matrix(X_audio_norm), tag_weight * X_rym_norm])


@app.get("/")
def home():
    return {"home is where we are not present"}

def recommend_similar(album_title, data, feature_cols, n=10):
    query = album_title.lower().strip()
    mask = data["album"].str.lower().str.contains(query, na=False)
    matches = data[mask]
    if matches.empty:
        return [{"error": "Album not found"}]

    idx = matches.index[0]
    row_pos = data.index.get_loc(idx)
    subset_X_combined = X_combined[data.index, :]
    sims = cosine_similarity(subset_X_combined[row_pos:row_pos + 1], subset_X_combined).flatten()
    sim_df = data.copy()
    sim_df["similarity"] = sims
    results = (
        sim_df.sort_values("similarity", ascending=False)
        .iloc[1:n+1]
        .apply(
            lambda row: {
                "artist": row["artist"],
                "album": row["album"],
                "pitchfork_score": row["score"],
                "rym_score": row.get("avg_rating", None),
                "similarity": row["similarity"]
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
