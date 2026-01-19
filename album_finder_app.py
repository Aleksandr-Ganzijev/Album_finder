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
# Normalize keys for a simple merge (IMPROVE MATCHING LATER),
# creates new columns which should be easier to match
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

# scaler.fit_transform makes it so everything become comparable, without it loudness and tempo overpower everything
# ^ Make all audio features speak the same numerical language
# normalize - each row/album vector becomes a point on a unit hypersphere for cos-similarity
scaler = StandardScaler()
numpy_spotify_audio_norm = normalize(scaler.fit_transform(numpy_spotify_audio))

# For every unique tag across albums create one column, value = TF-IDF weight
# A SciPy sparse matric is created (scr)
scr_rym = vectorizer.fit_transform(tag_text)

# normalize - each album's tag vector becomes a point on a unit hypersphere for cos-similarity;
# should compare distribution not quantity of tags
scr_rym_norm = normalize(scr_rym)

# csr_matrix = Put audio in the same container type as tags (convert dense audio matrix to sparse to match TF-IDF format)
# ^ numpy to SciPy
# weigh with the values we chose
# hstack concatenates columns, not rows
combined_csr = hstack([audio_weight * csr_matrix(numpy_spotify_audio_norm), tag_weight * scr_rym_norm])


# CSR MATRIX!!!!!!
# [0, 0, 0, 0.82, 0, 0.12]
# data = [0.82, 0.12]
# indices = [3, 5]
# indptr = [0, 2]
# All main, secondary genres are included, as well as descriptors from spotify and rym (columns)
# Each row represents an album


# Start a fastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"home is where we are not present"}

def recommend_similar(album_title, data, n=10):
    # Strip the query and lowercase it, lower for all albums in data
    query = album_title.lower().strip()
    albums_lower = data["album"].str.lower()

    # Search for matches, exact and starts produce dataframes
    # Matches = dataframe with all matches
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

    # Select the first row that matches, idx is the original df index, use it to pull the corresponding row from data
    # We need row_pos for reference to the corresponding csr matrix since csr does not have indexes, it has rows
    idx = matches.index[0]
    row_pos = data.index.get_loc(idx)

    # Data.index gives the row numbers of data; the csr matrix still behaves like a 2D array, you can index it
    # Using it to index combined_csr selects the corresponding rows in the csr matrix, ":" copies columns
    # "Take these specific rows", needed if the data is filtered, and we need the specific rows: score>=min_score
    subset_combined_csr = combined_csr[data.index, :]

    # Cosine similarity happens here, we need row_pos:row_pos+1 for a 2d array shape
    # The csr matrix still behaves like a 2D array, you can index it (1, n_rows)
    # .flatten() converts it to a 1D array (n_rows, ) so we can attach it as a column
    sims = cosine_similarity(subset_combined_csr[row_pos:row_pos + 1], subset_combined_csr).flatten()

    # Copy the dataframe and attach similarity to the new df as a column
    sim_df = data.copy()
    sim_df["similarity"] = sims

    # This checks the value and makes it parsable for JSON since NaN isn't decoded but None is
    def safe_nan(x):
        return None if pd.isna(x) or (isinstance(x, float) and math.isnan(x)) else x

    # This returns the results of each albums similarity to our query as a list of dictionaries
    # First sort by similarity in descending order
    # Then skip the most similar since it's the album itself and go until n, which is set by the user
    # .apply(axis=1) passes each row into the lambda, producing a pandas series of dicts
    # Extract the artist, album, pitchfork_score, rym_score and similarity from every other album
    # Since we .tolist() the pandas series where each element is a dictionary,
    # What returns is a list of dictionaries that is JSON-serializable and works better for manipulation in API
    results = (
        sim_df.sort_values("similarity", ascending=False)
        .iloc[1:n+1]
        .apply(
            lambda row: {
                "artist": row["artist"],
                "album": row["album"],
                "pitchfork_score": safe_nan(row["score"]),
                "rym_score": safe_nan(row["avg_rating"]),
                "similarity": safe_nan(row["similarity"]),
            },
            axis=1
        )
        .tolist()
    )
    return results

@app.get("/recommend")
def recommend(album: str, n: int = 10, min_score: float = 7.5):
    # Find the album in the full dataset, for the mask treat all NaNs as False
    # If masks numpy array contains anything by the match:
    # Return dictionary with the album and artist, else return as a dictionary
    # This structure makes it clear to the client the search failed and there are no recommendations to show
    # ^ FastAPI automatically converts python dictionaries into JSON responses
    mask = merged["album"].str.lower().str.contains(album.lower(), na=False)
    if mask.any():
        searched_album = merged[mask].iloc[0]
        heading = {
            "album": searched_album["album"],
            "artist": searched_album["artist"],
        }
    else:
        return {"searched": {"error": "Album not found"}, "recommendations": []}

    # Filter dataset to only recommend albums that have the min_score threshold
    filtered_df=merged[merged["score"]>=min_score]

    # Call recommend similar to generate recommendations from filtered_df
    # We return the heading with the album and artist as well as the list of dictionaries with similar albums
    recs = recommend_similar(searched_album["album"], filtered_df, n)
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