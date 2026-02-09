"""
OpenAI Integration for TV Show Recommender.

This module handles all AI-related features: generating fictional TV series
ideas from user favorites, creating DALL·E poster images, and assembling
the results into a DataFrame for display.

Dependencies:
    - openai: Chat completions (GPT) and image generation (DALL·E 2)
    - dotenv: Load OPENAI_API_KEY from .env file

Environment variables:
    OPENAI_API_KEY: Required for AI features
    OPENAI_ORGANIZATION, OPENAI_PROJECT: Optional, for org/project-scoped keys
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

# Load .env if present (works for both source and PyInstaller builds, as long as .env is next to the exe)
load_dotenv()


@lru_cache(maxsize=1)
def _get_openai_client():
    """
    Return an OpenAI client (lazy, only when AI features are used).
    This prevents the whole app from crashing at import time when no key exists.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY. Provide it in a .env file or as an environment variable.")

    org = os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT")
    kwargs = {"api_key": api_key}
    if org:
        kwargs["organization"] = org
    if project:
        kwargs["project"] = project

    return OpenAI(**kwargs)


def extract_title_and_description(text: str):
    """
    Parse GPT output to extract TV series name and description.

    Expects text in the format:
        TV Series name: <name>
        TV Series short description: <description>

    Args:
        text: Raw string from GPT response.

    Returns:
        Tuple of (title, description). Strips surrounding quotes.

    Raises:
        ValueError: If expected markers are not found in the output.
    """
    title_marker = "TV Series name: "
    desc_marker = "TV Series short description: "

    title_pos = text.find(title_marker)
    if title_pos == -1:
        raise ValueError(f"Expected '{title_marker}' not found in GPT output:\n{text[:200]}")
    title_start = title_pos + len(title_marker)
    title_end = text.find("\n", title_start)
    if title_end == -1:
        title_end = len(text)
    title = text[title_start:title_end].strip().strip('"')

    desc_pos = text.find(desc_marker)
    if desc_pos == -1:
        raise ValueError(f"Expected '{desc_marker}' not found in GPT output:\n{text[:200]}")
    desc_start = desc_pos + len(desc_marker)
    desc_end = text.find("\nTV Series name:", desc_start)  # next title (if exists)
    description = text[desc_start:desc_end if desc_end != -1 else None].strip().strip('"')

    return title, description


def create_tv_series_names_and_descriptions(initial_shows, recommended_shows):
    """
    Use GPT to create two fictional TV series: one based on user favorites,
    one based on the recommended shows.

    Args:
        initial_shows: List of show titles the user liked.
        recommended_shows: DataFrame or list of recommended show titles.

    Returns:
        Tuple of (raw_text_initial, raw_text_recommended) from GPT.
    """
    client = _get_openai_client()

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
    response_chat_initial = client.chat.completions.create(
        seed=1,
        messages=[{"role": "user", "content": prompt_template.format(shows=initial_shows)}],
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=250,
        timeout=60.0,
    )
    choice_initial = response_chat_initial.choices[0]
    if choice_initial.message.content is None:
        reason = getattr(choice_initial, "finish_reason", None)
        raise ValueError(
            f"Model returned no content (finish_reason={reason}). "
            "Expected marker not found; possibly refused or content-filtered."
        )
    response_chat_text_initial = choice_initial.message.content

    response_chat_recommended = client.chat.completions.create(
        seed=1,
        messages=[{"role": "user", "content": prompt_template.format(shows=recommended_titles)}],
        model="gpt-4o-mini",
        temperature=0.8,
        max_tokens=250,
        timeout=60.0,
    )
    choice_recommended = response_chat_recommended.choices[0]
    if choice_recommended.message.content is None:
        reason = getattr(choice_recommended, "finish_reason", None)
        raise ValueError(
            f"Model returned no content (finish_reason={reason}). "
            "Expected marker not found; possibly refused or content-filtered."
        )
    response_chat_text_recommended = choice_recommended.message.content

    return response_chat_text_initial, response_chat_text_recommended


def create_tv_series_photo(description: str):
    """
    Generate a TV-series poster image from a description using DALL·E 2.

    Args:
        description: Text description of the fictional TV series.

    Returns:
        URL string of the generated image (512x512).
    """
    client = _get_openai_client()

    # DALL·E 2 is cheaper than DALL·E 3, good enough for a small poster in this project
    response_image = client.images.generate(
        model="dall-e-2",
        prompt=f"Create a TV-series poster or wall art, based on this description: {description}",
        n=1,
        size="512x512",
        timeout=60.0,
    )

    return response_image.data[0].url


def create_ai_tv(initial_shows, recommended_shows):
    """
    Create two AI-generated fictional TV series with titles, descriptions, and poster images.

    Show #1 is based on the user's favorite shows; Show #2 is based on the
    recommended shows. Each gets a DALL·E-generated poster image.

    Args:
        initial_shows: List of show titles the user liked.
        recommended_shows: DataFrame or list of recommended show titles.

    Returns:
        DataFrame with columns: Title, Description, Image (URLs).
    """
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
