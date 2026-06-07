# Dataset Schema

This document describes the complete schema for all data files in the `traffic-monitor-lizard` repository. It covers both the **raw CSV files** stored in version control and the **derived columns** computed by the analysis toolkit (`data_utils.py`, `traffic_analyzer.py`).

---

## Overview

| File | Purpose | Storage |
|------|---------|---------|
| `csv-traffic-bangalore.csv` | Timestamped traffic readings (all historical) | Versioned |
| `csv-routes-bangalore.csv` | Route definitions and metadata | Versioned |
| `csv-locations_*.csv` | Location names mapped to Plus Codes | Versioned |
| `csv-weather-snapshot.csv` | Latest weather reading per route | Versioned |

The traffic file uses a **long format**: one row per route per reading. Each reading captures the estimated travel time from Google Maps at a specific moment.

---

## Raw Files

### `csv-traffic-bangalore.csv`

The primary dataset. Each row is one traffic snapshot for one route.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `date` | `date` (ISO 8601) | No | Calendar date of collection, e.g. `2025-09-25` |
| `time` | `time` (24-hour, `HH:MM`) | No | Collection time in local timezone, e.g. `14:25` |
| `route_code` | `string` | No | Origin and destination Plus Codes joined by `\|`, e.g. `XJG4+7J\|5PX4+HQ` |
| `duration` | `integer` (minutes) | No | Estimated travel time in minutes |
| `distance` | `float` (km) | No | Route distance in kilometres |
| `temp` | `integer` (°C) | Yes | Temperature at nearest weather station |
| `realfeel` | `integer` (°C) | Yes | "Real feel" temperature |
| `humidity` | `integer` (%) | Yes | Relative humidity percentage |
| `rsi_flag` | `string` | Yes | Rain / precipitation status, e.g. `Heavy Rain` or empty |
| `aqi` | `integer` | Yes | Air quality index value |

**Constraints**
- `duration` > 0
- `distance` > 0
- Rows with missing `duration` or `distance` are discarded by the scraper before writing

**Notes**
- Weather columns may be empty for early readings collected before weather integration was added
- `rsi_flag` is normalised to empty string when the raw value is `"No Precipitation"`
- The file is appended to by the GitHub Actions workflow on every successful scrape

---

### `csv-routes-bangalore.csv`

Metadata for each monitored route. One row per route.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `route_code` | `string` | No | Same identifier used in the traffic file; format: `{origin_plus_code}\|{destination_plus_code}` |
| `label_full` | `string` | No | Human-readable long name, e.g. `MG Road Metro Station → Kempegowda International Airport, Bengaluru` |
| `label_short` | `string` | No | Short display label, e.g. `Airport Expy` |
| `map_link` | `URL` | No | Direct Google Maps directions link |
| `accuweather_station` | `string` | No | AccuWeather station slug used for weather lookups, e.g. `shantala-nagar/3352203` |

**Primary Key**: `route_code`

---

### `csv-locations_{lat}_{lon}.csv`

Maps Plus Codes to human-readable place names. The reference latitude and longitude in the filename are used by the scraper to decode short Plus Codes.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `plus_code` | `string` | No | Open Location Code (Plus Code), e.g. `XJG4+7J` |
| `location` | `string` | No | Human-readable place name used in Google Maps queries |

**Primary Key**: `plus_code`

**Relationship to other files**
- `plus_code` appears as the origin or destination part of `route_code` in the routes and traffic files
- The scraper uses this file to translate Plus Codes into queryable place names for Google Maps

---

### `csv-weather-snapshot.csv`

Latest weather reading for each route, updated by the weather snapshot script before each traffic scrape.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `route_code` | `string` | No | Same route identifier |
| `route_name_short` | `string` | No | Short route label (duplicated for convenience) |
| `accuweather_station` | `string` | No | Weather station slug |
| `temp` | `integer` (°C) | No | Current temperature |
| `temp_flag` | `string` | No | Temperature qualifier, e.g. `Cloudy` |
| `realfeel` | `integer` (°C) | No | "Real feel" temperature |
| `realfeel_flag` | `string` | No | Real-feel qualifier, e.g. `Pleasant` |
| `humidity` | `integer` (%) | No | Relative humidity |
| `rsi_flag` | `string` | No | Rain status, e.g. `No Precipitation` |
| `rsi_forecast` | `string` | No | Short precipitation forecast text |
| `aqi` | `integer` | No | Air quality index |
| `aqi_flag` | `string` | No | AQI qualifier, e.g. `Fair` |

**Notes**
- This is overwritten in place on every weather snapshot run
- The traffic scraper reads this file and joins weather fields into each CSV row it outputs
- `rsi_flag` values of `"No Precipitation"` are normalised to empty string in the traffic output

---

## Derived Columns

When you load the traffic data via `data_utils.py` or `traffic_analyzer.py`, the following columns are computed automatically.

### From `preprocess_traffic_data()`

| Column | Type | Formula / Source |
|--------|------|-----------------|
| `avg_speed` | `float` (km/h) | `distance / (duration / 60)` |

### From `compute_temporal_features()`

| Column | Type | Formula / Source |
|--------|------|-----------------|
| `year` | `integer` | Parsed from `date` |
| `month` | `integer` | Parsed from `date` |
| `day` | `integer` | Parsed from `date` |
| `hour` | `integer` (0-23) | Parsed from `time` (`HH:MM`) |
| `timestamp` | `datetime64[ns]` | `pd.to_datetime(df[['year','month','day','hour']])` |
| `day_of_week` | `string` | `timestamp.dt.day_name()`, e.g. `Monday` |
| `is_weekend` | `boolean` | `timestamp.dt.dayofweek >= 5` |
| `time_category` | `Categorical` | Binned from `hour`: |
| | | `late_night` : 00:00 – 06:00 |
| | | `morning` : 06:00 – 08:00 |
| | | `morning_rush` : 08:00 – 11:00 |
| | | `early_afternoon` : 11:00 – 14:00 |
| | | `late_afternoon` : 14:00 – 18:00 |
| | | `evening_rush` : 18:00 – 21:00 |
| | | `night` : 21:00 – 24:00 |

---

## Data Relationships

```
csv-locations_{lat}_{lon}.csv
    plus_code ───┐
                 │
    csv-routes-bangalore.csv
        route_code = origin_plus_code | destination_plus_code
                 │
    csv-traffic-bangalore.csv
        route_code ───┘
        (weather fields copied from csv-weather-snapshot.csv at scrape time)
```

- **Locations ↔ Routes**: A route's `route_code` is built from two `plus_code` values in the locations file
- **Routes ↔ Traffic**: `route_code` is the foreign key linking traffic readings to route metadata
- **Routes ↔ Weather**: `accuweather_station` is shared between routes and the weather snapshot
- **Weather → Traffic**: Weather values are denormalised into the traffic CSV at scrape time; the snapshot file is transient

---

## Quality Rules

Rules enforced by the scraper (`traffic_snapshot.py`) before writing:

1. Durations must be parseable (handles `"25 min"`, `"1 hr 5 min"`, `"2 hr"`, `"7 min"`)
2. Distances must contain `" km"`; the suffix is stripped and converted to float
3. Any row missing both a valid `duration` and a valid `distance` is dropped

Rules enforced by `preprocess_traffic_data()` when loading:

1. Deduplicate on `(route_code, date, hour, duration, distance)` — keeps first occurrence
2. Drop rows with missing `route_code`, `duration`, `distance`, or `avg_speed`
3. Sort by `(year, month, day, hour)`

---

## File Naming Convention

| Pattern | Meaning |
|---------|---------|
| `csv-traffic-{city}.csv` | Traffic readings for a specific city |
| `csv-routes-{city}.csv` | Route definitions for a specific city |
| `csv-locations_{lat}_{lon}.csv` | Location lookup for a city; lat/lon are the reference coordinates used for Plus Code decoding |
| `csv-weather-snapshot.csv` | Single latest weather reading per route (overwritten on every run) |

The current instance covers **Bangalore, India**.
