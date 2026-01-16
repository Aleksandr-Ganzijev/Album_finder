from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# Load data (copy-paste your cleaning code)
pd.set_option("display.width", None)
df = pd.read_csv("output-data.csv")
feature_cols = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "tempo"]
df = df.dropna(subset=["album", "artist", "score"] + feature_cols)
X = df[feature_cols].to_numpy()
app = FastAPI()

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
    X_local = data[feature_cols].to_numpy()
    sims = cosine_similarity(X_local[row_pos:row_pos + 1], X_local)[0]
    sim_df = data.copy()
    sim_df["similarity"] = sims
    results = (sim_df.sort_values("similarity", ascending=False)
               .iloc[1:n+1][["artist", "album", "score", "similarity"]].to_dict("records"))
    return results

@app.get("/recommend")
def recommend(album: str, n: int = 10, min_score: float = 7.5):
    filtered_df = df[df["score"]>=min_score]
    recs = recommend_similar(album, filtered_df, feature_cols, n)
    mask = filtered_df["album"].str.lower().str.contains(album.lower(), na=False)
    if mask.any():
        searched_album = filtered_df[mask].iloc[0]
        heading ={
            "album": searched_album['album'],
            'artist': searched_album['artist']
        }
    else:
        heading = {"error": "Album not found"}
    return {"searched":heading,"recommendations":recs}
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
    <button onclick="getRecommendations()">Recommend</button>
    <h2 id="searched-album-heading"></h2>
    <ul id="results"></ul>

    <script>
        async function getRecommendations() {
            const album = document.getElementById('album').value;
            const heading = document.getElementById('searched-album-heading');

            const minScore = document.getElementById('min_score').value;
            const res = await fetch(
                `/recommend?album=${encodeURIComponent(album)}&min_score=${minScore}`
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
                    li.textContent = `${item.artist} - ${item.album} (Score: ${item.score}, Similarity: ${item.similarity.toFixed(2)})`;
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
