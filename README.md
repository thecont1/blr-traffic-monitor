# Traffic Monitor Lizard

The data engine behind [traffiCOracle](https://github.com/thecont1/traffic-oracle). This project builds a dataset of live hyperlocal traffic and weather readings, and provides a full toolkit for analysing and visualising the results.

It is designed for **civic technologists**, **urban planners**, **data journalists**, and **researchers** who want a transparent, reproducible pipeline for understanding how a city moves, how traffic patterns evolve, what factors influence travel times, and the characteristics of roads and routes.

---

## What it does

An automated script periodically asks Google Maps to estimate how long it would take to go from Point A to Point B, for each from a set of pre-determined routes, and records the result. Over weeks and months, this builds up a rich dataset. The project also pulls in local weather data so you can later ask whether rain or heat correlate with traffic congestion.

Because everything is stored as plain CSV files in a public GitHub repository, the data is **open** (anyone can download and verify it), **versioned** (every commit is a snapshot in time) and **reusable** (the companion dashboard traffiCOracle reads these files directly).

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Data](#data)
- [How it works](#how-it-works)
- [License](#license)

---

## Features

- **Hands-free data collection**  
  A single command launches a headless Chrome browser, visits Google Maps for every configured route, and outputs clean CSV rows. No API keys, no paid services. This repo will always be ready for you with data less than 20 minutes old.

- **30+ built-in visualisations**  
  Hourly heatmaps, time-series decomposition, radar charts, ranking animations, control charts for anomaly detection, travel-time reliability curves, and forecast plots with confidence intervals.

- **Interactive notebooks**  
  Two Jupyter notebooks let you explore the data without writing scripts from scratch: one for quick visual exploration, one for comprehensive worked examples.

- **R³S² scoring**  
  The proprietary *Rolling Relative Route Scoring System* ranks routes by reliability and speed relative to each other. The system is validated with correlation tests, sensitivity-to-outliers checks, and stability-over-time analysis to certify trustworthiness.

- **Statistical rigour**  
  Built-in tests for normality, stationarity, autocorrelation, and variance homogeneity. Outlier detection with IQR, Z-score, or isolation-forest methods.

- **Weather correlation**  
  Each traffic snapshot is paired with temperature, "real feel", humidity, rain status, and air-quality index from a local weather station. Stored alongside the route data for later analysis.

---

## Quick Start

### Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| **Python** | Runtime (3.13+) | [python.org](https://python.org) or `brew install python` |
| **uv** | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Chrome** | Browser for scraping | [google.com/chrome](https://www.google.com/chrome/) |

### 1. Install dependencies

```bash
git clone https://github.com/thecont1/traffic-monitor-lizard.git
cd traffic-monitor-lizard
uv sync
```

`uv sync` reads `pyproject.toml` and installs everything in one step.

### 2. Collect fresh traffic data

```bash
uv run python tools/traffic_snapshot.py
```

This will:
- Launch a headless Chrome window
- Query Google Maps for each route
- Print CSV rows to your terminal
- Print a one-line commit summary as the last line

To **append** the output to the historical dataset:

```bash
uv run python tools/traffic_snapshot.py >> data/csv-traffic.csv
```

### 3. Explore the data in a notebook

```bash
# Quick visual exploration
uv run jupyter notebook traffic_visual.ipynb

# Comprehensive analysis examples
uv run jupyter notebook traffic_analysis_examples.ipynb
```

### 4. Run the tests

```bash
uv run pytest tests/
```

### Useful commands

```bash
uv run python tools/weather.py --json        # Collect a fresh weather snapshot
uv run python tools/fix_timestamps.py --apply # Deduplicate the traffic CSV
uv run pytest tests/ --cov=. --cov-report=html # Run tests with coverage
```

---

## Data

### Files overview

| File | Purpose |
|------|---------|
| `data/csv-traffic-*.csv` | All timestamped traffic readings (city-specific) |
| `data/csv-routes-*.csv` | Route definitions and display metadata (city-specific) |
| `data/csv-locations_*.csv` | Location names mapped to Plus Codes |
| `data/csv-weather-snapshot.csv` | Latest weather snapshot for each route |

### Traffic data columns

| Column | Example | Meaning |
|--------|---------|---------|
| `date` | `2025-09-25` | Calendar date of collection |
| `time` | `14:25` | Collection time (24-hour, local time) |
| `route_code` | `2HM2+P8\|XJV5+RG` | Origin and destination Plus Codes joined by `\|` |
| `duration` | `32` | Travel time in minutes |
| `distance` | `11.0` | Route distance in kilometres |
| `temp` | `24` | Temperature in °C |
| `realfeel` | `23` | "Real feel" temperature in °C |
| `humidity` | `77` | Relative humidity percentage |
| `rsi_flag` | `Heavy Rain` | Rain / precipitation status |
| `aqi` | `96` | Air quality index value |

### Routes data columns

| Column | Example | Meaning |
|--------|---------|---------|
| `route_code` | `XJG4+7J\|5PX4+HQ` | Same identifier used in the traffic file |
| `label_full` | `Origin → Destination` | Human-readable long name |
| `label_short` | `Airport Expy` | Short display label |
| `map_link` | `https://maps.app.goo.gl/...` | Direct Google Maps link |
| `accuweather_station` | `shantala-nagar/3352203` | Weather station used for this route |

### Data quality

Before rows are written, the scraper already performs basic sanity checks:

- Missing or unparsable durations are discarded
- Distances are stripped of the `" km"` suffix and converted to numbers
- Any row lacking both a valid distance and a valid duration is dropped

The companion Python module `data_utils.py` adds further cleaning when you load the data for analysis:

- Removes duplicates (same route, same day, same hour)
- Computes `avg_speed` from `distance / (duration / 60)`
- Adds temporal features: `timestamp`, `day_of_week`, `is_weekend`, `time_category` (late_night, morning, morning_rush, early_afternoon, late_afternoon, evening_rush, night)

---

## How it works

### The pipeline

```
cron-job.org  ──►  Cloudflare Worker  ──►  GitHub Actions
 (twice/hour)        (secret check)         (ubuntu runner)
                                              │
                                              ▼
                                        Chrome + Selenium
                                              │
                                              ▼
                                        Google Maps queries
                                              │
                                              ▼
                                        CSV rows appended
                                              │
                                              ▼
                                        Commit to GitHub
                                              │
                                              ▼
                              traffiCOracle reads the public CSV
```

1. **Trigger** — An external scheduler or manual dispatch triggers the GitHub Actions workflow.
2. **Scraping** — A runner installs `uv`, launches Chrome, and runs `traffic_snapshot.py`.
3. **Storage** — New rows are appended to the traffic CSV and committed. If the remote has diverged (e.g. concurrent workflow runs), the runner automatically rebases and retries with `--force-with-lease` so no data is lost.
4. **Consumption** — traffiCOracle (or your own script) fetches the updated CSV from GitHub's raw-content URL.

### Deduplication

A daily deduplication pass keeps one reading per route per hour in the historical dataset. Recent data (< 24 hours) is left untouched so the current day's analysis benefits from higher-frequency readings. The dedup is idempotent and commits the cleaned file with a count of removed records.

## Licensing

This repository contains two separately licensed parts:

- **Code**: [GNU AGPL v3.0](https://github.com/thecont1/traffic-monitor-lizard/blob/main/LICENSE_CODE.md)
- **Dataset**: [CC BY 4.0](https://github.com/thecont1/traffic-monitor-lizard/blob/main/LICENSE_DATA.md)

The code and data are intentionally licensed separately so the data can remain openly reusable while the collection engine stays open and forkable.
