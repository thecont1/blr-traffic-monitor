"""
FastAPI application — ephemeral traffic analysis service.

Lifespan loads the dataset once at startup.  Health endpoints prove
the process is alive (/livez) and that data is loaded (/readyz).
All analysis routes live under /api/.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from service.config import settings
from service.data_loader import Dataset, load_dataset

logger = logging.getLogger("traffic-service")

# Module-level reference — set during lifespan, read by routers.
_dataset: Dataset | None = None


def get_dataset() -> Dataset:
    """Return the loaded dataset.  Raises RuntimeError if not loaded."""
    if _dataset is None:
        raise RuntimeError("Dataset not loaded — service is not ready")
    return _dataset


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dataset on startup; yield; nothing to clean up."""
    global _dataset
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info("Starting traffic analysis service (mode=%s)", settings.traffic_mode)

    try:
        _dataset = load_dataset(settings.data_dir, settings.city_config)
        logger.info("Dataset loaded — service is ready")
    except Exception:
        logger.exception("Failed to load dataset")
        _dataset = None

    # Write openapi.json to output dir for agent consumption
    _write_openapi_artifact(app)

    yield  # service runs here

    logger.info("Shutting down")


def _write_openapi_artifact(app: FastAPI) -> None:
    """Persist the OpenAPI schema to /output/openapi.json on disk."""
    try:
        out = settings.output_dir / "openapi.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        schema = app.openapi()
        out.write_text(json.dumps(schema, indent=2))
        logger.info("OpenAPI schema written to %s", out)
    except Exception:
        logger.warning("Could not write openapi.json artifact", exc_info=True)


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Traffic Monitor Lizard — Analysis API",
    description=(
        "Ephemeral analysis service for Bangalore traffic data. "
        "Provides route enumeration, traffic reads, aggregation, comparison, "
        "percentile/distribution analysis, anomaly detection, and report generation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Health endpoints ────────────────────────────────────────────────

@app.get("/livez", tags=["health"])
async def livez():
    """Liveness: the process is up."""
    return {"status": "alive"}


@app.get("/readyz", tags=["health"])
async def readyz():
    """Readiness: dataset is loaded and the service can answer queries."""
    ds = _dataset
    if ds is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "dataset not loaded"},
        )
    return {
        "status": "ready",
        "routes": ds.route_count,
        "rows": ds.row_count,
        "city": ds.city_config.get("display_name", "unknown"),
    }


# ── Register routers ───────────────────────────────────────────────

from service.routers import anomalies, compare, reports, routes, stats, traffic  # noqa: E402

app.include_router(routes.router, prefix="/api")
app.include_router(traffic.router, prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(anomalies.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
