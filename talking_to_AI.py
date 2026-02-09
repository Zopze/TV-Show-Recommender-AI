
import openai
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_title_and_description(text):
    title_start = text.find("TV Series name: ") + len("TV Series name: ")
    title_end = text.find('\n', title_start)
    title = text[title_start:title_end].strip().strip('"')  # Strips whitespace and extra quotes

    desc_start = text.find("TV Series short description: ") + len("TV Series short description: ")
    desc_end = text.find('\nTV Series name:', desc_start)  # next title
    description = text[desc_start:desc_end if desc_end != -1 else None].strip().strip('"')

    return title, description

def create_tv_series_names_and_descriptions(initial_shows, recommended_shows):

    # Generate TV series names and descriptions for initial shows
    response_chat_initial = openai.ChatCompletion.create(
        seed=1,
        messages=[
            {
                'role': 'user',
                'content': f'''You are a creative TV Series Creator-Writer, 
                                based on this list of TV shows: {initial_shows}, 
                                create a new TV series name and a short IMDb-like description
                                TV Series name:
                                TV Series short description:'''
            }
        ],
        model="gpt-3.5-turbo",
    )

    response_chat_text_initial = response_chat_initial['choices'][0]['message']['content']

    # Generate TV series names and descriptions for recommended shows
    response_chat_recommended = openai.ChatCompletion.create(
        seed=1,
        messages=[
            {
                'role': 'user',
                'content': f'''You are a creative TV Series Creator-Writer, 
                                based on this list of TV shows: {recommended_shows}, 
                                create a new TV series name and a short IMDb-like description
                                TV Series name:
                                TV Series short description:'''
            }
        ],
        model="gpt-3.5-turbo",
    )

    response_chat_text_recommended = response_chat_recommended['choices'][0]['message']['content']

    return response_chat_text_initial, response_chat_text_recommended

def create_tv_series_photo(description):
        response_image = openai.Image.create(
            model='dall-e-3',
            prompt=f'Create a TV-series poster or wall art, based on this description: {description}',
            n=1,
            size='1024x1024',  # or '256x256' or '512x512' or '1024x1024' or '2048x2048'
            quality='standard',
        )

        return response_image['data'][0]['url']

def create_ai_tv(initial_shows, recommended_shows):
    response_from_initial_shows, response_from_recommended_shows = create_tv_series_names_and_descriptions(initial_shows, recommended_shows)
    titles = []
    descriptions = []
    image_urls = []

    title_initial, description_initial = extract_title_and_description(response_from_initial_shows)

    titles.append(title_initial)
    descriptions.append(description_initial)

    title_recommended, description_recommended = extract_title_and_description(response_from_recommended_shows)

    titles.append(title_recommended)
    descriptions.append(description_recommended)

    image_url_initial = create_tv_series_photo(description_initial)
    image_urls.append(image_url_initial)

    image_url_recommended = create_tv_series_photo(description_recommended)
    image_urls.append(image_url_recommended)


    generated_shows = pd.DataFrame({
        'Title': titles,
        'Description': descriptions,
        'Image': image_urls
    })
    return generated_shows