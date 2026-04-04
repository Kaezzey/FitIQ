from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from api.store import (
    DEFAULT_RANKINGS_PATH,
    DEFAULT_SUMMARY_PATH,
    RankingStore,
    build_ranking_store,
)

API_VERSION = "1.0.0"


class HealthResponse(BaseModel):
    status: str
    api_version: str
    artifact_version: str
    artifact_generated_at_utc: str | None = None
    artifact_loaded_at_utc: str
    score_column: str | None = None
    query_group_column: str
    total_ranked_products: int
    query_group_count: int
    docs_enabled: bool


class QueryGroupItem(BaseModel):
    query_group_v2: str
    product_count: int


class QueryGroupsResponse(BaseModel):
    total_query_groups: int
    items: list[QueryGroupItem]


class RankingItem(BaseModel):
    asin: str
    parent_asin: str | None = None
    category: str | None = None
    subcategory: str
    ranking_group: str
    query_group_v2: str
    score_fitiq_v1: float
    rank_in_query_group: int
    query_group_size: int
    rank_percentile: float
    review_count: int
    mean_rating: float


class RankingsResponse(BaseModel):
    query_group_v2: str
    total_products: int
    limit: int
    offset: int
    items: list[RankingItem]


class ProductResponse(RankingItem):
    pass


def get_store(request: Request) -> RankingStore:
    return request.app.state.ranking_store


def get_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def create_app(
    rankings_path: Path = DEFAULT_RANKINGS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    show_docs: bool | None = None,
) -> FastAPI:
    docs_enabled = get_env_bool("FITIQ_API_EXPOSE_DOCS", False) if show_docs is None else bool(show_docs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ranking_store = build_ranking_store(
            rankings_path=rankings_path,
            summary_path=summary_path,
        )
        yield

    app = FastAPI(
        title="FitIQ Ranking API",
        version=API_VERSION,
        description="Read-only API serving the FitIQ v1 ranked product artifact.",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.docs_enabled = docs_enabled

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        store = get_store(request)
        payload = store.health_payload()
        payload["api_version"] = API_VERSION
        payload["docs_enabled"] = bool(request.app.state.docs_enabled)
        return HealthResponse(**payload)

    @app.get("/query-groups", response_model=QueryGroupsResponse)
    async def list_query_groups(request: Request) -> QueryGroupsResponse:
        store = get_store(request)
        items = [QueryGroupItem(**item) for item in store.list_query_groups()]
        return QueryGroupsResponse(total_query_groups=len(items), items=items)

    @app.get("/rankings/{query_group_v2}", response_model=RankingsResponse)
    async def get_rankings(
        query_group_v2: str,
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> RankingsResponse:
        store = get_store(request)
        try:
            payload = store.get_rankings(query_group_v2=query_group_v2, limit=limit, offset=offset)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown query_group_v2: {query_group_v2}") from exc
        items = [RankingItem(**item) for item in payload["items"]]
        return RankingsResponse(
            query_group_v2=payload["query_group_v2"],
            total_products=payload["total_products"],
            limit=payload["limit"],
            offset=payload["offset"],
            items=items,
        )

    @app.get("/products/{asin}", response_model=ProductResponse)
    async def get_product(asin: str, request: Request) -> ProductResponse:
        store = get_store(request)
        try:
            payload = store.get_product(asin)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown ASIN: {asin}") from exc
        return ProductResponse(**payload)

    return app


app = create_app()
