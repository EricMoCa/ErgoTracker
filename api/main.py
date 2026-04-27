from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import settings
from .routes import analysis, rules, reports, setup


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.reports_output_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ErgoTracker API started")
    yield
    logger.info("ErgoTracker API stopped")


app = FastAPI(
    title="ErgoTracker API",
    description="Post-process ergonomic analysis from monocular video",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router)
app.include_router(rules.router)
app.include_router(reports.router)
app.include_router(setup.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
