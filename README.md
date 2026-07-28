# Banking Semantic Search System

An end-to-end semantic search application for a bank's customer-support query archive. Powered by **FastText subword n-gram embeddings**, **Word2Vec**, and **FastAPI**, with a premium bank-ledger themed web interface.

---

## 🏛️ Features

- **Subword Semantic Search**: Handles typos, informal phrasing, and out-of-vocabulary (OOV) words (e.g., `"acount balanc"` → correctly matches `Account Management` queries).
- **Side-by-Side Model Comparison**: Real-time evaluation of **FastText (Subword N-Grams)** vs. **Word2Vec (Whole Words Only)**.
- **Intent Category Prediction**: Automatic Top-1 nearest-neighbor classification into 5 banking domains.
- **Category & Threshold Filtering**: Restrict search to specific categories or filter results by minimum similarity threshold.
- **Feedback Collection System**: Built-in `/api/feedback` endpoint and UI buttons (👍 / 👎) to capture match quality feedback.
- **Bank Ledger UI**: Responsive single-page web app built with Vanilla HTML/CSS/JS, featuring parchment cards, rotated brass percentage match stamps, and staggered animations.

---

## 📁 Repository Structure

```
banking-semantic-search/
├── main.py                     # FastAPI application: REST endpoints, CORS, lifespan
├── search_engine.py            # BankingSemanticSearch engine class & dual model loader
├── train_models.py             # Dataset generator (451 queries) & model training script
├── requirements.txt            # Python dependencies
├── .gitignore
├── static/
│   └── index.html              # Bank ledger UI (Single Page Application)
├── models/
│   ├── fasttext_banking.model  # Trained FastText model
│   ├── word2vec_banking.model  # Trained Word2Vec model
│   ├── train_vectors_fasttext.npy # Pre-computed reference vectors
│   └── train.csv               # Reference training dataset
├── data/
│   ├── banking_semantic_search_dataset.csv # Full dataset
│   ├── train.csv, val.csv, test.csv         # Stratified 80/10/10 splits
│   └── search_feedback.jsonl                # User feedback storage
└── tests/
    ├── test_search_engine.py           # Engine & vectorization unit tests
    └── test_comparison_and_feedback.py  # API endpoint tests
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+ or Python 3.12

### 2. Installation
```bash
# Clone repository
git clone https://github.com/surjeetjothi/Banking-semantic-search.git
cd Banking-semantic-search

# Create & activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Training (Optional - pre-trained models included)
```bash
python train_models.py
```

### 4. Run Unit Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

### 5. Launch Application
```bash
uvicorn main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

## 📊 Dataset & Categories

~451 natural-language banking customer questions stratified across 5 categories:
- **`Account Management`** (98 queries)
- **`Loans`** (94 queries)
- **`Card Services`** (92 queries)
- **`Payments & Transfers`** (92 queries)
- **`Fraud & Security`** (75 queries)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web UI |
| `GET` | `/api/health` | Model status, vocab size, category list |
| `GET` | `/api/categories` | Returns available categories |
| `POST` | `/api/search` | Performs FastText semantic search |
| `POST` | `/api/compare` | Compares FastText vs Word2Vec side-by-side |
| `POST` | `/api/feedback` | Saves search match feedback |

---

## 📜 License
MIT License
