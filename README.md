# ShowSuggesterAI — TV Show Recommender (Embeddings + Vectors) 🎬🤖

A Python project from **“Exercise 2 — The magic of Embeddings and Vectors”** (Software Development Using AI – Assignment #2).  
The program asks you for TV shows you loved, **matches them with fuzzy search**, builds an **average embedding vector**, and recommends the **top 5 most similar shows** using **cosine similarity**.

It also supports generating **two custom “new shows”** and **two TV-show ads** using OpenAI (ChatGPT + DALL·E), as described in the exercise.

---

## ✨ What this project does (required flow)

1. Ask the user to enter multiple TV shows separated by commas.
2. Use **fuzzy string matching** to map user input → real show titles, then confirm with the user (y/n).  
3. Generate recommendations:  
   - Load embeddings from disk (pickle)
   - Compute the **average vector** of the input shows
   - Find the **5 closest** shows (excluding the input shows)
   - Print them with a “match %” score.
4. Create:
   - **Show #1** based on the user’s input shows
   - **Show #2** based on the recommended shows
   - **2 ads** (images) for the shows using DALL·E. 

---

## 🧠 How recommendations work (high level)

- **Dataset:** `imdb_tvshows.csv` (provided by the course).  
- **Embeddings:** we embed each show’s *Description* once, save `{title -> vector}` to a pickle file, and reuse it on every run (to save cost + time). 
- **Similarity:** cosine similarity between the user’s average vector and every show vector.
- **Fuzzy matching:** `thefuzz.process.extractOne` (Levenshtein distance) to map typos to real show titles.  

---

## 📦 Releases (Windows `.exe`)

If you want to use the app without installing Python:

1. Go to this repo’s **Releases**
2. Download the latest **Windows build** ZIP (contains the `.exe`)
3. Extract the ZIP
4. (Optional) Set your API key if you want the AI-generated parts
5. Run the `.exe`

### 🔐 API key is never included
This project **does not ship any keys**.  
If you want the OpenAI features (show creation / ads), you must provide **your own** API key.

**Option A — `.env` file (recommended):**
1. Copy `.env.example` → `.env`
2. Fill:
   ```env
   OPENAI_API_KEY="your_key_here"
   ```

**Option B — Environment variable (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_key_here"
.\ShowSuggesterAI.exe
```

If no key is provided, the program should still run the **recommendation logic** (depending on the current implementation), but AI generation will be disabled.

---

## ▶️ Run from source (Python)

### Requirements
- Python **3.10+** recommended
- pip

### 1) Clone
```bash
git clone https://github.com/dorhaboosha/TV-Show-Recommender-AI.git
cd TV-Show-Recommender-AI
```

### 2) Create virtual environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Configure environment (optional)
```bash
cp .env.example .env
```
Then edit `.env` and add your key (if needed).

### 5) Run
```bash
python ShowSuggesterAI.py
```

---

## 🧪 Tests (TDD)

The exercise requires developing with **TDD** (tests first → red → green → refactor).
Run tests (depending on your setup):

- If you use `pytest`:
```bash
pytest -q
```

- Or run the included test script:
```bash
python ShowSuggesterAI_Test.py
```

### Coverage (recommended)
```bash
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

---

## 🗂️ Project files

- `ShowSuggesterAI.py` — main entry point (CLI flow)
- `embedding_file.py` — embedding load/save + vector utilities
- `talking_to_AI.py` — OpenAI prompts (show creation + ads)
- `imdb_tvshows.csv` — dataset
- `imdb_tvshows_embedding.pkl` — cached embeddings dictionary (pickle)
- `ShowSuggesterAI_Test.py` — tests

---