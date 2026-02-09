"""
Embedding Generator and Loader for TV Show Recommender.

This module handles the creation and loading of text embeddings for TV shows
from the IMDB dataset. Embeddings are generated using OpenAI's text-embedding-ada-002
model, combining genre and description for each show to enable semantic search
and similarity-based recommendations.

Dependencies:
    - openai: For generating embeddings via OpenAI API
    - pandas: For reading the TV shows CSV
    - dotenv: For loading environment variables (OPENAI_API_KEY)

Input:
    - imdb_tvshows.csv: CSV with columns 'Title', 'Description', 'Genres'

Output:
    - imdb_tvshows_embedding.pkl: Pickle file mapping show titles to embedding vectors
"""

import openai
from dotenv import load_dotenv
import os
import pickle
import pandas as pd

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load TV shows dataset
df = pd.read_csv("imdb_tvshows.csv")

# Dictionary mapping show titles to their embedding vectors
embed_dict = {}
for i, row in df.iterrows():
    title = row["Title"]
    description = row["Description"]
    genre = row["Genres"]
    text = genre + " - " + description

    # --- Embedding generation (commented out - uses API credits) ---
    # Uncomment the block below to regenerate embeddings from scratch. Requires OPENAI_API_KEY.
    """
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002',
    )

    embed_dict[title] = response['data'][0]['embedding']

with open('imdb_tvshows_embedding.pkl', 'wb') as f:
    pickle.dump(embed_dict, f)"""

# Load pre-computed embeddings for use in recommendations
with open('imdb_tvshows_embedding.pkl', 'rb') as f:
    info = pickle.load(f)
