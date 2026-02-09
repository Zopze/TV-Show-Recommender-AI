
from ShowSuggesterAI import automatic_translator, ai_recommendation, show_image
import pandas as pd
from unittest.mock import patch

def test_automatic_translator():
    val = 'valid value'
    assert automatic_translator(None, None) == []
    assert automatic_translator(None, "") == []
    assert automatic_translator("", None) == []
    assert automatic_translator(val, None) == []
    assert automatic_translator(None, val) == []
    assert automatic_translator([], val) == []
    assert automatic_translator([], None) == []

    data = {
        'Title': ['Game Of Thrones', 'Lupin', 'The Witcher', 'How I Met Your Mother', 'Friends', 'Brooklyn Nine-Nine',
                 'Stranger Things', 'Riverdale']
    }
    df = pd.DataFrame(data)
    assert automatic_translator(['game of thrones', 'lupin', 'riverdale'], df) == ['Game Of Thrones', 'Lupin', 'Riverdale']
    assert automatic_translator(['Lopin', 'Rivedale', 'frid'], df) == ['Lupin', 'Riverdale', 'Friends']
    assert automatic_translator(['howi metyou', 'watcher', 'strange thing', 'brook 99'], df) == ['How I Met Your Mother', 'The Witcher', 'Stranger Things', 'Brooklyn Nine-Nine']

def test_Ai_recommendation():
    df = pd.DataFrame({
        'Title': ['How I Met Your Mother', 'The Witcher', 'Stranger Things'],
        'Embedding': [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    })
    shows_list = []
    recommend_shows, generate_shows = ai_recommendation(shows_list, df)
    assert recommend_shows.empty
    assert isinstance(generate_shows, pd.DataFrame) and generate_shows.empty

    shows_list = ['How I Met Your Mother', 'The Witcher']
    embed_dict = {
        'How I Met Your Mother': [0.1, 0.2],
        'The Witcher': [0.2, 0.3],
        'Stranger Things': [0.3, 0.4]
    }

    with patch('ShowSuggesterAI.pickle.load', return_value=embed_dict):
        recommend_shows, generate_shows = ai_recommendation(shows_list, df)

    assert isinstance(recommend_shows, pd.DataFrame)
    assert isinstance(generate_shows, pd.DataFrame)

def test_show_image():
    # Assume you have invalid image urls
    df = pd.DataFrame({
        'Image': ['url1', 'url2']
    })

    try:
        show_image(df)
        assert True
    except:
        assert False

    # Assume you have valid image urls
    df = pd.DataFrame({
        'Image': ['https://incubator.ucf.edu/wp-content/uploads/2023/07/artificial-intelligence-new-technology-science-futuristic-abstract-human-brain-ai-technology-cpu-central-processor-unit-chipset-big-data-machine-learning-cyber-mind-domination-generative-ai-scaled-1.jpg',
                  'https://images.spiceworks.com/wp-content/uploads/2022/02/14135111/shutterstock_1154457493.jpg']
    })

    try:
        show_image(df)
        assert True
    except:
        assert False




