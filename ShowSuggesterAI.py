
from thefuzz import fuzz
import pandas as pd
import numpy as np
import pickle
from openai.embeddings_utils import cosine_similarity
from talking_to_AI import create_ai_tv
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def show_image(df):
    # Checking that Image column exists in the DataFrame
    if 'Image' not in df.columns:
        raise ValueError("DataFrame does not has an 'Image' column")

    # Checking that there are at least two rows for two shows
    if len(df) < 2:
        raise ValueError("DataFrame needs at least two rows to display images")

    image_urls = [df.loc[0, 'Image'], df.loc[1, 'Image']]

    # Setting the size of the figure
    plt.figure(figsize=(10, 5))  # Adjust the size

    # Display the images
    for i, image_url in enumerate(image_urls):
        ax = plt.subplot(1, 2, i + 1)

        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            ax.imshow(image)
            ax.axis('off')

        except Exception as e:
            holder_image = 'error-message.png'
            image = Image.open(holder_image)
            ax.imshow(image)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    return image_urls

def automatic_translator(shows_list, df):
    if shows_list is None or shows_list == [] or df is None:
        return []
    correct_shows_list = []
    for show in shows_list:
        df['Ratio'] = df['Title'].apply(lambda title: fuzz.ratio(show, title))
        max_ratio_row = df.loc[df['Ratio'].idxmax()]
        correct_shows_list.append(max_ratio_row['Title'])
    return correct_shows_list


def ai_recommendation(shows_list, df):
    if not shows_list:
        return pd.DataFrame(), pd.DataFrame()

    with open('imdb_tvshows_embedding.pkl','rb') as f:
        embed_dict_info = pickle.load(f)

    df['Embedding'] = df['Title'].apply(lambda title: embed_dict_info[title])

    # Compute the average embedding vector of all shows in given shows list
    avg_embed = np.mean([embed_dict_info[show] for show in shows_list if show in embed_dict_info], axis=0)

    # Function that compute the similarity with the average embedding.
    def computing_similarity(row):
        return cosine_similarity(np.array(row), avg_embed)

    # Computing the similarity of each show's embedding with the average embedding.
    df['Similarity'] = df['Embedding'].apply(computing_similarity)

    # Creating a DataFrame that contains the shows from the file without the shows the user gave.
    df_without_input_shows = df[~df['Title'].isin(shows_list)]

    # Sorting the DataFrame that based on similarity.
    df_sort = df_without_input_shows.sort_values('Similarity', ascending=False)

    #The recomnended Shows that we found in the file.
    recommendation_shows = df_sort.head(5)

    #The shows that the AI create.
    generate_shows = create_ai_tv(shows_list, recommendation_shows)

    return recommendation_shows, generate_shows

if __name__ == '__main__':
    df = pd.read_csv('imdb_tvshows.csv')
    tv_shows = []
    correction = 'n'
    while True:
        tv_shows = input("Which TV shows did you love watching?"
                         "\nSeparate them by a comma and Make sure to enter more than 1 show\n")
        tv_shows = tv_shows.split(',')
        if len(tv_shows) <= 1:
            print("Please enter more than 1 show, Separated by a commas.\n")
            continue

        correct_shows = automatic_translator(tv_shows, df)
        correction = input(f"Just to make sure, do you mean {correct_shows}? (y/n)\n")

        if correction != 'y':
            print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly")
        else:
            print("Great! Generating recommendations…")
            break

    recommendation_shows, generate_shows = ai_recommendation(correct_shows, df)
    print("Here are the tv shows that i think you would love:\n")

    for i, row in recommendation_shows.iterrows():
        print(f"{row['Title']} ({row['Similarity']*100:.0f}%)\n")

    print(f"I have also created just for you two shows which I think you would love."
          f"\nShow #1 is based on the fact that you loved the input shows that you gave me."
          f"\nIts name is {generate_shows.at[0, 'Title']} and it is about {generate_shows.at[0, 'Description']}"
          f"\nShow #2 is based on the shows that I recommended for you. "
          f"\nIts name is {generate_shows.at[1, 'Title']} and it is about {generate_shows.at[1, 'Description']}"
          f"\nHere are also the 2 tv show ads. Hope you like them!")
    show_image(generate_shows)

