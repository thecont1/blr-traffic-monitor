#!/usr/bin/env python3
"""
Generate 1050×1050 route mapshots for TraffiCOracle.

Each route gets a PNG with the route line centered on an OpenStreetMap base.
Uses OSRM demo server for routing and standard OSM tile server.

Output: frontend/public/mapshots/<route_slug>_1050.png
"""

import csv
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from openlocationcode import openlocationcode as olc

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = 1050
TILE_SIZE = 256
TILE_ZOOM = 13  # good for Bangalore city-level routes
ROUTE_COLOR = (255, 60, 60, 220)  # semi-transparent red
ROUTE_WIDTH = 6
OUTLINE_COLOR = (255, 255, 255, 180)
OUTLINE_WIDTH = 10
PADDING_PX = 80  # padding around the route bounding box

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "traffiCOracle-mapshot-generator/1.0"

REF_LAT = 12.9514242
REF_LNG = 77.6590212

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR.parent / "traffiCOracle" / "public" / "mapshots"


# ── Tile math ───────────────────────────────────────────────────────────────
def lat_lng_to_tile(lat, lng, zoom):
    """Convert lat/lng to tile x/y at given zoom."""
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def lat_lng_to_pixel(lat, lng, zoom):
    """Convert lat/lng to pixel position in tile coordinates."""
    n = 2 ** zoom
    x = (lng + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


# ── OSRM routing ────────────────────────────────────────────────────────────
def fetch_route_coords(origin, destination):
    """Fetch route polyline from OSRM. Returns list of (lng, lat) or None."""
    coords_str = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
    url = OSRM_URL.format(coords=coords_str)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("code") == "Ok" and data.get("routes"):
            return data["routes"][0]["geometry"]["coordinates"]  # [[lng, lat], ...]
    except Exception as e:
        print(f"  OSRM error: {e}", file=sys.stderr)
    return None


# ── Tile fetching with caching ──────────────────────────────────────────────
tile_cache = {}


def fetch_tile(x, y, z):
    """Fetch a single OSM tile image. Uses in-memory cache."""
    key = (z, x, y)
    if key in tile_cache:
        return tile_cache[key]
    url = TILE_URL.format(z=z, x=x, y=y)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        tile_cache[key] = img
        return img
    except Exception as e:
        print(f"  Tile fetch error ({z}/{x}/{y}): {e}", file=sys.stderr)
        # Return blank tile
        blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (240, 240, 240, 255))
        tile_cache[key] = blank
        return blank


# ── Map rendering ───────────────────────────────────────────────────────────
def render_map(center_lat, center_lng, zoom, width, height):
    """Render a map image centered on lat/lng at given zoom and pixel size."""
    center_px = lat_lng_to_pixel(center_lat, center_lng, zoom)
    cx, cy = center_px

    # Tile range we need
    min_px_x = cx - width / 2
    min_px_y = cy - height / 2
    max_px_x = cx + width / 2
    max_px_y = cy + height / 2

    min_tx = int(min_px_x // TILE_SIZE)
    min_ty = int(min_px_y // TILE_SIZE)
    max_tx = int(max_px_x // TILE_SIZE)
    max_ty = int(max_px_y // TILE_SIZE)

    canvas = Image.new("RGBA", (width, height), (240, 240, 240, 255))

    for tx in range(min_tx, max_tx + 1):
        for ty in range(min_ty, max_ty + 1):
            tile = fetch_tile(tx, ty, zoom)
            paste_x = int(tx * TILE_SIZE - min_px_x)
            paste_y = int(ty * TILE_SIZE - min_px_y)
            canvas.paste(tile, (paste_x, paste_y))

    return canvas, center_px, (min_px_x, min_px_y)


def draw_route(canvas, route_coords, zoom, center_px, origin_offset):
    """Draw the route polyline on the canvas."""
    draw = ImageDraw.Draw(canvas)
    cx, cy = center_px
    ox, oy = origin_offset

    # Convert route coordinates to pixel positions on canvas
    points = []
    for lng, lat in route_coords:
        px, py = lat_lng_to_pixel(lat, lng, zoom)
        points.append((int(px - ox), int(py - oy)))

    if len(points) < 2:
        return

    # Draw white outline for contrast
    draw.line(points, fill=OUTLINE_COLOR, width=OUTLINE_WIDTH, joint="curve")
    # Draw route line
    draw.line(points, fill=ROUTE_COLOR, width=ROUTE_WIDTH, joint="curve")

    # Draw origin and destination markers
    marker_r = 10
    if points:
        # Origin: green circle
        o = points[0]
        draw.ellipse([o[0]-marker_r, o[1]-marker_r, o[0]+marker_r, o[1]+marker_r],
                     fill=(34, 197, 94, 230), outline=(255,255,255,200), width=3)
        # Destination: red circle
        d = points[-1]
        draw.ellipse([d[0]-marker_r, d[1]-marker_r, d[0]+marker_r, d[1]+marker_r],
                     fill=(239, 68, 68, 230), outline=(255,255,255,200), width=3)


def compute_route_bbox(route_coords):
    """Get bounding box of route coordinates."""
    lngs = [c[0] for c in route_coords]
    lats = [c[1] for c in route_coords]
    return min(lats), max(lats), min(lngs), max(lngs)


def compute_zoom_for_bbox(lat_min, lat_max, lng_min, lng_max, img_size, padding):
    """Find the best zoom level to fit the route bbox in the image."""
    for zoom in range(16, 10, -1):
        # Check if bbox fits in the image at this zoom
        px_min = lat_lng_to_pixel(lat_max, lng_min, zoom)  # top-left (note lat inversion)
        px_max = lat_lng_to_pixel(lat_min, lng_max, zoom)  # bottom-right
        w = px_max[0] - px_min[0]
        h = px_max[1] - px_min[1]
        if w < (img_size - 2 * padding) and h < (img_size - 2 * padding):
            return zoom
    return 11


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    # Read locations
    locations = {}
    loc_file = DATA_DIR / "csv-locations_12.9514242_77.6590212.csv"
    with open(loc_file) as f:
        for row in csv.DictReader(f):
            pc = row["plus_code"].strip()
            full = olc.recoverNearest(pc, REF_LAT, REF_LNG)
            area = olc.decode(full)
            locations[pc] = (area.latitudeCenter, area.longitudeCenter)

    # Read routes
    routes = []
    route_file = DATA_DIR / "csv-routes-bangalore.csv"
    with open(route_file) as f:
        for row in csv.DictReader(f):
            rc = row["route_code"]
            label = row["label_short"].strip()
            parts = rc.split("|")
            if len(parts) == 2 and parts[0] in locations and parts[1] in locations:
                routes.append({
                    "route_code": rc,
                    "label_short": label,
                    "origin": locations[parts[0]],
                    "destination": locations[parts[1]],
                })

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating mapshots for {len(routes)} routes...")
    print(f"Output: {OUT_DIR}")
    print()

    for i, route in enumerate(routes):
        label = route["label_short"]
        slug = label.lower().replace(" ", "_").replace("'", "")
        out_path = OUT_DIR / f"{slug}_1050.png"

        if out_path.exists():
            print(f"[{i+1}/{len(routes)}] {label}: already exists, skipping")
            continue

        print(f"[{i+1}/{len(routes)}] {label}...")

        # Fetch route from OSRM
        route_coords = fetch_route_coords(route["origin"], route["destination"])
        if not route_coords:
            print(f"  FAILED: no OSRM route")
            continue

        # Compute bbox and zoom
        lat_min, lat_max, lng_min, lng_max = compute_route_bbox(route_coords)
        center_lat = (lat_min + lat_max) / 2
        center_lng = (lng_min + lng_max) / 2
        zoom = compute_zoom_for_bbox(lat_min, lat_max, lng_min, lng_max, IMG_SIZE, PADDING_PX)

        print(f"  zoom={zoom}, center=({center_lat:.4f}, {center_lng:.4f}), points={len(route_coords)}")

        # Render map
        canvas, center_px, origin_offset = render_map(center_lat, center_lng, zoom, IMG_SIZE, IMG_SIZE)

        # Draw route
        draw_route(canvas, route_coords, zoom, center_px, origin_offset)

        # Save
        canvas_rgb = canvas.convert("RGB")
        canvas_rgb.save(str(out_path), "PNG", optimize=True)
        size_kb = out_path.stat().st_size / 1024
        print(f"  Saved: {out_path.name} ({size_kb:.0f} KB)")

        # Rate limit: be nice to tile servers
        time.sleep(0.5)

    print(f"\nDone! Mapshots in {OUT_DIR}")


if __name__ == "__main__":
    main()
