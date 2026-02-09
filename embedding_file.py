"""
Embedding Loader for TV Show Recommender.

Loads pre-computed text embeddings from imdb_tvshows_embedding.pkl for use in
similarity-based recommendations. To regenerate embeddings, iterate over
imdb_tvshows.csv and use OpenAI's text-embedding-ada-002 model.
"""

import pickle

# --- Embedding generation code removed ---
# To regenerate embeddings, iterate over imdb_tvshows.csv and for each row create
# text = row["Genres"] + " - " + row["Description"], then call:
#   openai.Embedding.create(input=text, model='text-embedding-ada-002')
# and dump the {title: embedding} dict to imdb_tvshows_embedding.pkl.


def load_embeddings(path: str = 'imdb_tvshows_embedding.pkl') -> dict:
    """
    Load pre-computed embeddings from a pickle file.

    Args:
        path: Path to the pickle file (default: imdb_tvshows_embedding.pkl).

    Returns:
        Dict mapping show titles to embedding vectors.
    """
    with open(path, 'rb') as f:
        # Trusted, project-internal artifact (Ruff S301)
        return pickle.load(f)
