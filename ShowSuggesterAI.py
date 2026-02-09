"""
Show Suggester AI - TV Show Recommendation Engine.

Core module for the TV Show Recommender that handles:
- Fuzzy matching of user input to show titles (automatic_translator)
- Similarity-based recommendations using embeddings (ai_recommendation)
- Display of show images or AI-generated artwork (show_image)

Dependencies:
    - thefuzz: Fuzzy string matching for show name correction
    - pandas, numpy: Data handling and similarity computation
    - talking_to_AI: OpenAI integration for AI-generated show suggestions
    - PIL, matplotlib: Image display

Data files required:
    - imdb_tvshows.csv: TV show catalog
    - imdb_tvshows_embedding.pkl: Pre-computed embeddings
    - error-message.png: Fallback image when URLs fail
"""

from thefuzz import fuzz
import pandas as pd
import numpy as np
from embedding_file import load_embeddings
from talking_to_AI import create_ai_tv
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os
import sys


def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def cosine_similarity(a, b) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: vector-like (list/np.ndarray)
        b: vector-like (list/np.ndarray)

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if one vector is all zeros.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def show_image(df):
    """
    Display side-by-side images for the first two shows in the DataFrame.

    Fetches images from URLs in the 'Image' column. On fetch failure, shows
    a placeholder (error-message.png). Opens a matplotlib window.

    Args:
        df: DataFrame with an 'Image' column containing URLs.

    Returns:
        List of the two image URLs that were displayed.

    Raises:
        ValueError: If 'Image' column is missing or DataFrame has fewer than 2 rows.
    """
    # Checking that Image column exists in the DataFrame
    if 'Image' not in df.columns:
        raise ValueError("DataFrame does not have an 'Image' column")

    # Checking that there are at least two rows for two shows
    if len(df) < 2:
        raise ValueError("DataFrame needs at least two rows to display images")

    image_urls = [df.iloc[0]['Image'], df.iloc[1]['Image']]

    # Setting the size of the figure
    plt.figure(figsize=(10, 5))

    # Display the images
    for i, image_url in enumerate(image_urls):
        ax = plt.subplot(1, 2, i + 1)

        try:
            response = requests.get(image_url, timeout=20)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            ax.imshow(image)
            ax.axis('off')
        except Exception:
            # Fallback placeholder image (make sure this file exists in your project)
            holder_image = resource_path('error-message.png')
            try:
                image = Image.open(holder_image)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Fallback image '{holder_image}' not found. Add error-message.png to your project."
                ) from None
            ax.imshow(image)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    return image_urls


def automatic_translator(shows_list, df):
    """
    Map user input (possibly misspelled) to correct show titles from the catalog.

    Uses fuzzy string matching (thefuzz) to find the best match for each
    user-entered show name against the DataFrame's Title column.

    Args:
        shows_list: List of show names as entered by the user.
        df: DataFrame with a 'Title' column (catalog of known shows).

    Returns:
        List of corrected/matched show titles. Empty list if input is invalid.
    """
    if not shows_list or df is None or not isinstance(df, pd.DataFrame):
        return []

    correct_shows_list = []
    for show in shows_list:
        ratios = df['Title'].apply(lambda title, _show=show: fuzz.ratio(_show, title))
        max_ratio_row = df.loc[ratios.idxmax()]
        correct_shows_list.append(max_ratio_row['Title'])

    return correct_shows_list


def ai_recommendation(shows_list, df):
    """
    Recommend TV shows based on input favorites using embedding similarity.

    Computes the average embedding of the user's favorite shows, then finds
    the 5 most similar shows from the catalog. Optionally calls OpenAI to
    generate fictional show ideas and display artwork.

    Args:
        shows_list: List of show titles the user likes (already matched by automatic_translator).
        df: DataFrame with 'Title' column (must match keys in embedding pickle).

    Returns:
        Tuple of:
        - recommendation_shows (DataFrame): Top 5 recommended shows with Similarity column.
        - generate_shows (DataFrame): AI-generated shows/ads (empty if OpenAI fails).
    """
    if not shows_list:
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()

    embed_dict_info = load_embeddings(resource_path('imdb_tvshows_embedding.pkl'))

    df['Embedding'] = df['Title'].apply(lambda title: embed_dict_info.get(title))
    df = df[df['Embedding'].notna()]

    valid_embeds = [embed_dict_info[show] for show in shows_list if show in embed_dict_info]
    if not valid_embeds:
        return pd.DataFrame(), pd.DataFrame()

    avg_embed = np.mean(valid_embeds, axis=0)

    # Function that compute the similarity with the average embedding.
    def computing_similarity(row):
        return cosine_similarity(np.array(row), avg_embed)

    # Computing the similarity of each show's embedding with the average embedding.
    df['Similarity'] = df['Embedding'].apply(computing_similarity)

    # Creating a DataFrame that contains the shows from the file without the shows the user gave.
    df_without_input_shows = df[~df['Title'].isin(shows_list)]

    # Sorting the DataFrame based on similarity.
    df_sort = df_without_input_shows.sort_values('Similarity', ascending=False)

    # The recommended shows that we found in the file.
    recommendation_shows = df_sort.head(5)

    # The shows that the AI creates (optional)
    try:
        generate_shows = create_ai_tv(shows_list, recommendation_shows)
    except Exception as e:
        print("\n[AI features disabled] OpenAI call failed, continuing without AI-generated shows/ads.")
        print("Reason:", str(e))
        generate_shows = pd.DataFrame()  # empty DF so the program can continue safely

    return recommendation_shows, generate_shows


if __name__ == '__main__':
    # Interactive CLI: prompts for favorite shows, shows recommendations and AI-generated content
    df = pd.read_csv(resource_path('imdb_tvshows.csv'))

    while True:
        tv_shows = input(
            "Which TV shows did you love watching?"
            "\nSeparate them by a comma and make sure to enter more than 1 show\n"
        )

        # Split + trim spaces around each title
        tv_shows = [s.strip() for s in tv_shows.split(',') if s.strip()]

        if len(tv_shows) <= 1:
            print("Please enter more than 1 show, separated by commas.\n")
            continue

        correct_shows = automatic_translator(tv_shows, df)
        correction = input(f"Just to make sure, do you mean {correct_shows}? (y/n)\n").strip().lower()

        if correction != 'y':
            print("\nSorry about that. Let's try again — please make sure to write the show names correctly.\n")
        else:
            print("\nGreat! Generating recommendations…")
            break

    recommendation_shows, generate_shows = ai_recommendation(correct_shows, df)

    print("\nHere are the TV shows that I think you would love:\n")
    for _, row in recommendation_shows.iterrows():
        sim = max(0.0, row['Similarity'])
        print(f"{row['Title']} ({sim * 100:.0f}%)\n")

    # Only show AI-generated content if it exists
    if not generate_shows.empty and len(generate_shows) >= 2:
        print(
            "I have also created just for you two shows which I think you would love."
            "\nShow #1 is based on the fact that you loved the input shows that you gave me."
            f"\nIts name is {generate_shows.iloc[0]['Title']} and it is about {generate_shows.iloc[0]['Description']}"
            "\nShow #2 is based on the shows that I recommended for you."
            f"\nIts name is {generate_shows.iloc[1]['Title']} and it is about {generate_shows.iloc[1]['Description']}"
            "\nHere are also the 2 TV show ads. Hope you like them!"
        )

        # If your AI output DF contains 'Image' URLs, show them
        try:
            show_image(generate_shows)
        except Exception as e:
            print("\nCould not display images:", str(e))
    else:
        print("\nAI-generated shows/ads were skipped (OpenAI API not available).")
        print("You can still use the recommendations normally.")
    
    # Keep the console open when running as a PyInstaller EXE (double-click)
    if getattr(sys, "frozen", False):
        input("\nPress Enter to exit...")