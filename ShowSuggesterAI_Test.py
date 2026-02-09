"""
Test suite for ShowSuggesterAI module.

Tests the core components of the TV Show Recommender:
- automatic_translator: Fuzzy matching of user input to show titles
- ai_recommendation: Recommendation logic with empty/valid inputs
- show_image: Image display handling with valid/invalid URLs

Run with: pytest ShowSuggesterAI_Test.py -v
"""

from ShowSuggesterAI import automatic_translator, ai_recommendation, show_image
import pandas as pd
from unittest.mock import patch, MagicMock

def test_automatic_translator():
    """Test fuzzy matching of user input to show titles in the dataset."""
    val = 'valid value'
    # Edge cases: empty or None inputs return empty list
    assert automatic_translator(None, None) == []
    assert automatic_translator(None, "") == []
    assert automatic_translator("", None) == []
    assert automatic_translator(val, None) == []
    assert automatic_translator(None, val) == []
    assert automatic_translator([], val) == []
    assert automatic_translator([], None) == []

    # Fuzzy matching: misspellings and partial names resolve to correct titles
    data = {
        'Title': ['Game Of Thrones', 'Lupin', 'The Witcher', 'How I Met Your Mother', 'Friends', 'Brooklyn Nine-Nine',
                 'Stranger Things', 'Riverdale']
    }
    df = pd.DataFrame(data)
    assert automatic_translator(['game of thrones', 'lupin', 'riverdale'], df) == ['Game Of Thrones', 'Lupin', 'Riverdale']
    assert automatic_translator(['Lopin', 'Rivedale', 'frid'], df) == ['Lupin', 'Riverdale', 'Friends']
    assert automatic_translator(['howi metyou', 'watcher', 'strange thing', 'brook 99'], df) == ['How I Met Your Mother', 'The Witcher', 'Stranger Things', 'Brooklyn Nine-Nine']

def test_ai_recommendation():
    """Test recommendation logic with empty input and valid show lists."""
    df = pd.DataFrame({
        'Title': ['How I Met Your Mother', 'The Witcher', 'Stranger Things'],
        'Embedding': [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    })
    # Empty input returns empty DataFrames
    shows_list = []
    recommend_shows, generate_shows = ai_recommendation(shows_list, df)
    assert recommend_shows.empty
    assert isinstance(generate_shows, pd.DataFrame) and generate_shows.empty

    # Valid input with mocked embeddings returns non-empty DataFrames
    shows_list = ['How I Met Your Mother', 'The Witcher']
    embed_dict = {
        'How I Met Your Mother': [0.1, 0.2],
        'The Witcher': [0.2, 0.3],
        'Stranger Things': [0.3, 0.4]
    }

    with patch('ShowSuggesterAI.load_embeddings', return_value=embed_dict), \
         patch('builtins.open', MagicMock()):
        recommend_shows, generate_shows = ai_recommendation(shows_list, df)

    assert isinstance(recommend_shows, pd.DataFrame)
    assert isinstance(generate_shows, pd.DataFrame)
    assert not recommend_shows.empty, "Valid input should return recommendations"
    # Input shows are excluded; Stranger Things is the only remaining show
    assert 'Stranger Things' in recommend_shows['Title'].values

def test_show_image():
    """Test show_image handles both invalid and valid image URLs."""
    expected_urls = ['url1', 'url2']
    df = pd.DataFrame({'Image': expected_urls})
    mock_ax = MagicMock()
    with patch('ShowSuggesterAI.requests.get', side_effect=Exception("network")), \
         patch('ShowSuggesterAI.plt.show'), \
         patch('ShowSuggesterAI.Image.open', return_value=MagicMock()), \
         patch('ShowSuggesterAI.plt.subplot', return_value=mock_ax), \
         patch('ShowSuggesterAI.plt.figure'):
        result = show_image(df)
    assert result == expected_urls
    assert result == df['Image'].tolist()

    # Valid URLs: mock network and image load to keep test hermetic
    expected_urls = ['https://example.com/img1.jpg', 'https://example.com/img2.jpg']
    df = pd.DataFrame({'Image': expected_urls})
    mock_response = MagicMock()
    mock_response.content = b'fake'
    mock_response.raise_for_status = MagicMock()
    mock_ax = MagicMock()
    with patch('ShowSuggesterAI.requests.get', return_value=mock_response), \
         patch('ShowSuggesterAI.plt.show'), \
         patch('ShowSuggesterAI.Image.open', return_value=MagicMock()), \
         patch('ShowSuggesterAI.plt.subplot', return_value=mock_ax), \
         patch('ShowSuggesterAI.plt.figure'):
        result = show_image(df)
    assert result == expected_urls
    assert result == df['Image'].tolist()




