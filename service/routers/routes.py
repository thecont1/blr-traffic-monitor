"""
Route enumeration — list all monitored routes with metadata.
"""

from fastapi import APIRouter

from service.main import get_dataset
from service.models import RouteInfo, RouteList

router = APIRouter(tags=["routes"])


@router.get("/routes", response_model=RouteList)
async def list_routes():
    """List all monitored routes with labels, map links, and weather stations."""
    ds = get_dataset()
    rows = ds.routes_df.to_dict(orient="records")
    return RouteList(routes=[RouteInfo(**r) for r in rows], count=len(rows))
