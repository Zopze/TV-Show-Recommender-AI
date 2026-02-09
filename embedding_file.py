
import openai
from dotenv import load_dotenv
import os
import pickle
import pandas as pd

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
df = pd.read_csv("imdb_tvshows.csv")

embed_dict = {}
for i, row in df.iterrows():
    title = row["Title"]
    description = row["Description"]
    genre = row["Genres"]
    text = genre + " - " + description

"""  The once calling for OpenAI about embedding
#   Get embeddings from the OpenAI
    response = openai.Embedding.create(
        input=text,
        model='text-embedding-ada-002',
    )

    embed_dict[title] = response['data'][0]['embedding']

with open('imdb_tvshows_embedding.pkl', 'wb') as f:
    pickle.dump(embed_dict, f)"""

with open('imdb_tvshows_embedding.pkl', 'rb') as f:
    info = pickle.load(f)
