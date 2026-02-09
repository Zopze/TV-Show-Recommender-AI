import os
from dotenv import load_dotenv
import pandas as pd
import openai

# Load .env if present (works for both source and PyInstaller builds, as long as .env is next to the exe)
load_dotenv()


def _configure_openai():
    """
    Configure OpenAI lazily (only when AI features are used).
    This prevents the whole app from crashing at import time when no key exists.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY. Provide it in a .env file or as an environment variable.")

    openai.api_key = api_key

    # Optional: if you ever decide to use organization/project, keep them optional.
    # If they're not set, OpenAI will use your default.
    org = os.getenv("OPENAI_ORGANIZATION")
    if org:
        openai.organization = org

    project = os.getenv("OPENAI_PROJECT")
    if project:
        # Some setups support project-scoped keys; harmless if unused.
        # If your OpenAI package doesn't support this property, it will be ignored.
        try:
            openai.project = project
        except Exception:
            pass


def extract_title_and_description(text: str):
    title_start = text.find("TV Series name: ") + len("TV Series name: ")
    title_end = text.find("\n", title_start)
    title = text[title_start:title_end].strip().strip('"')

    desc_start = text.find("TV Series short description: ") + len("TV Series short description: ")
    desc_end = text.find("\nTV Series name:", desc_start)  # next title (if exists)
    description = text[desc_start:desc_end if desc_end != -1 else None].strip().strip('"')

    return title, description


def create_tv_series_names_and_descriptions(initial_shows, recommended_shows):
    _configure_openai()

    # recommended_shows can be a DataFrame; send only a small list to the model
    recommended_titles = (
        recommended_shows["Title"].head(5).tolist()
        if isinstance(recommended_shows, pd.DataFrame) and "Title" in recommended_shows.columns
        else recommended_shows
    )

    prompt_template = """You are a creative TV Series Creator-Writer.
Based on this list of TV shows: {shows}
Create ONE new TV series.

Return EXACTLY in this format (2 lines only):
TV Series name: <name>
TV Series short description: <description>
"""

    # For your project: gpt-4o-mini is a solid balance of quality + cost
    response_chat_initial = openai.ChatCompletion.create(
        seed=1,
        messages=[{"role": "user", "content": prompt_template.format(shows=initial_shows)}],
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=250,
    )
    response_chat_text_initial = response_chat_initial["choices"][0]["message"]["content"]

    response_chat_recommended = openai.ChatCompletion.create(
        seed=1,
        messages=[{"role": "user", "content": prompt_template.format(shows=recommended_titles)}],
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=250,
    )
    response_chat_text_recommended = response_chat_recommended["choices"][0]["message"]["content"]

    return response_chat_text_initial, response_chat_text_recommended


def create_tv_series_photo(description: str):
    _configure_openai()

    # DALL·E 2 is cheaper than DALL·E 3, good enough for a small poster in this project
    response_image = openai.Image.create(
        model="dall-e-2",
        prompt=f"Create a TV-series poster or wall art, based on this description: {description}",
        n=1,
        size="512x512",
    )

    return response_image["data"][0]["url"]


def create_ai_tv(initial_shows, recommended_shows):
    response_from_initial_shows, response_from_recommended_shows = create_tv_series_names_and_descriptions(
        initial_shows, recommended_shows
    )

    titles, descriptions, image_urls = [], [], []

    title_initial, description_initial = extract_title_and_description(response_from_initial_shows)
    titles.append(title_initial)
    descriptions.append(description_initial)

    title_recommended, description_recommended = extract_title_and_description(response_from_recommended_shows)
    titles.append(title_recommended)
    descriptions.append(description_recommended)

    image_urls.append(create_tv_series_photo(description_initial))
    image_urls.append(create_tv_series_photo(description_recommended))

    return pd.DataFrame({"Title": titles, "Description": descriptions, "Image": image_urls})
