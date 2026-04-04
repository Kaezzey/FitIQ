from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import create_app
from api.store import DEFAULT_RANKINGS_PATH, DEFAULT_SUMMARY_PATH, build_ranking_store


def main() -> int:
    store = build_ranking_store()

    health = store.health_payload()
    assert health["status"] == "ok"
    assert "rankings_path" not in health
    assert "summary_path" not in health
    assert health["artifact_version"] == "product_rankings_v1"
    assert health["query_group_column"] == "query_group_v2"
    assert health["total_ranked_products"] > 0
    assert health["query_group_count"] > 0

    docs_off_app = create_app(show_docs=False)
    docs_off_routes = sorted(route.path for route in docs_off_app.routes)
    assert "/health" in docs_off_routes
    assert "/query-groups" in docs_off_routes
    assert "/rankings/{query_group_v2}" in docs_off_routes
    assert "/products/{asin}" in docs_off_routes
    assert "/static" not in docs_off_routes
    assert "/docs" not in docs_off_routes

    docs_on_app = create_app(show_docs=True)
    docs_on_routes = sorted(route.path for route in docs_on_app.routes)
    assert "/docs" in docs_on_routes

    query_groups = store.list_query_groups()
    assert query_groups, "query group list must not be empty"

    first_query_group = query_groups[0]["query_group_v2"]
    rankings_page = store.get_rankings(first_query_group, limit=5, offset=0)
    assert rankings_page["query_group_v2"] == first_query_group
    assert rankings_page["total_products"] >= len(rankings_page["items"])
    assert rankings_page["items"], "first query-group page must contain items"
    observed_ranks = [item["rank_in_query_group"] for item in rankings_page["items"]]
    assert observed_ranks == sorted(observed_ranks), "items must stay sorted by rank"

    empty_page = store.get_rankings(first_query_group, limit=5, offset=10_000_000)
    assert empty_page["items"] == [], "oversized offset should yield an empty page"

    first_asin = rankings_page["items"][0]["asin"]
    product = store.get_product(first_asin)
    assert product["asin"] == first_asin
    assert product["query_group_v2"] == first_query_group

    try:
        store.get_rankings("definitely_unknown_query_group", limit=5, offset=0)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown query_group_v2 should raise KeyError")

    try:
        store.get_product("definitely_unknown_asin")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown ASIN should raise KeyError")

    print("API smoke tests passed.")
    print(f"  total_ranked_products: {health['total_ranked_products']}")
    print(f"  query_group_count: {health['query_group_count']}")
    print(f"  sample_query_group: {first_query_group}")
    print(f"  sample_first_asin: {first_asin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
