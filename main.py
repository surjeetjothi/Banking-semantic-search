"""
Banking Semantic Search — FastAPI Application (with Comparison Mode & Feedback)

Endpoints:
    GET  /              → serves the frontend (static/index.html)
    GET  /api/health    → model status, vocab size, categories
    GET  /api/categories → list of categories
    POST /api/search    → FastText semantic search
    POST /api/compare   → FastText vs Word2Vec side-by-side search comparison
    POST /api/feedback  → User feedback submission endpoint
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from search_engine import load_all_engines, BankingSemanticSearch

# ---------------------------------------------------------------------------
# Logging & Directories
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("banking_search")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
FEEDBACK_FILE = os.path.join(DATA_DIR, "search_feedback.jsonl")

# ---------------------------------------------------------------------------
# Lifespan — Load FastText & Word2Vec models
# ---------------------------------------------------------------------------
engines: dict[str, BankingSemanticSearch] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engines
    logger.info("Starting up — loading FastText & Word2Vec engines …")
    try:
        engines = load_all_engines()
        logger.info("Engines loaded successfully.")
    except FileNotFoundError as e:
        logger.warning(f"Model files missing ({e}). Auto-running model training pipeline...")
        import train_models
        train_models.main()
        engines = load_all_engines()
        logger.info("Engines loaded successfully after training.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Banking Semantic Search API",
    description="Semantic search over banking customer-support queries with FastText & Word2Vec comparison.",
    version="1.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    category: str | None = Field(None, description="Filter by category (optional)")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class SearchResult(BaseModel):
    id: int
    query: str
    category: str
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    model_type: str
    predicted_category: str
    results: list[SearchResult]
    result_count: int
    oov_tokens: list[str]
    processing_time_ms: float


class CompareResponse(BaseModel):
    query: str
    fasttext: SearchResponse
    word2vec: SearchResponse


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    vocab_size: dict[str, int]
    reference_count: int | None = None
    categories: list[str] = []


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1)
    result_id: int
    result_query: str
    category: str
    feedback: str = Field(..., description="'helpful' or 'unhelpful'")
    comments: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_single_search(engine: BankingSemanticSearch, request: SearchRequest) -> SearchResponse:
    start = time.perf_counter()
    if request.category:
        results_df, oov_tokens = engine.search_by_category(
            query=request.query, category=request.category, top_k=request.top_k
        )
    else:
        results_df, oov_tokens = engine.search(
            query=request.query, top_k=request.top_k, min_score=request.min_score
        )

    predicted_cat = engine.predict_category(request.query)
    elapsed_ms = (time.perf_counter() - start) * 1000

    results = [
        SearchResult(
            id=int(row["id"]),
            query=row["query"],
            category=row["category"],
            similarity_score=round(float(row["similarity_score"]), 4),
        )
        for _, row in results_df.iterrows()
    ]

    return SearchResponse(
        query=request.query,
        model_type=engine.model_type,
        predicted_category=predicted_cat,
        results=results,
        result_count=len(results),
        oov_tokens=oov_tokens,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    if not engines:
        return HealthResponse(status="error", models_loaded=[])
    
    ft = engines.get("fasttext")
    w2v = engines.get("word2vec")
    
    return HealthResponse(
        status="ok",
        models_loaded=list(engines.keys()),
        vocab_size={
            "fasttext": len(ft.model.wv) if ft else 0,
            "word2vec": len(w2v.model.wv) if w2v else 0,
        },
        reference_count=len(ft.reference_df) if ft else 0,
        categories=sorted(ft.reference_df["category"].unique().tolist()) if ft else [],
    )


@app.get("/api/categories", response_model=list[str])
async def list_categories():
    ft = engines.get("fasttext")
    if not ft:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return sorted(ft.reference_df["category"].unique().tolist())


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    ft = engines.get("fasttext")
    if not ft:
        raise HTTPException(status_code=503, detail="Search engine not loaded.")
    
    res = run_single_search(ft, request)
    logger.info(
        f"SEARCH │ model=fasttext │ query={request.query!r} │ category={request.category} │ "
        f"results={res.result_count} │ predicted={res.predicted_category} │ latency={res.processing_time_ms}ms"
    )
    return res


@app.post("/api/compare", response_model=CompareResponse)
async def compare(request: SearchRequest):
    ft = engines.get("fasttext")
    w2v = engines.get("word2vec")
    if not ft or not w2v:
        raise HTTPException(status_code=503, detail="Both engines must be loaded for comparison.")

    ft_res = run_single_search(ft, request)
    w2v_res = run_single_search(w2v, request)

    logger.info(
        f"COMPARE │ query={request.query!r} │ fasttext_predicted={ft_res.predicted_category} │ "
        f"word2vec_predicted={w2v_res.predicted_category} │ oov_w2v={w2v_res.oov_tokens}"
    )

    return CompareResponse(
        query=request.query,
        fasttext=ft_res,
        word2vec=w2v_res,
    )


@app.post("/api/feedback")
async def submit_feedback(fb: FeedbackRequest):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": fb.query,
        "result_id": fb.result_id,
        "result_query": fb.result_query,
        "category": fb.category,
        "feedback": fb.feedback,
        "comments": fb.comments,
    }

    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"FEEDBACK │ query={fb.query!r} │ result_id={fb.result_id} │ type={fb.feedback}")
        return {"status": "success", "message": "Feedback recorded. Thank you!"}
    except Exception as e:
        logger.error(f"Failed to write feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback.")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
