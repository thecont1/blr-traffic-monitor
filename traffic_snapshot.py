import sys
import numpy as np
import pandas as pd
import os, time, re, urllib.parse
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from openlocationcode import openlocationcode as olc
from timezonefinder import TimezoneFinder
global locations_data, referenceLatitude, referenceLongitude
locations_data = "csv-locations_12.9514242_77.6590212.csv"
root = os.path.splitext(locations_data)[0]
_, referenceLatitude, referenceLongitude = root.split("_")
referenceLatitude = float(referenceLatitude)
referenceLongitude = float(referenceLongitude)
locations_df = pd.read_csv(locations_data)
routes_df = pd.read_csv("csv-routes.csv")
out_file = "csv-bangalore_traffic"
tf = TimezoneFinder()

@lru_cache(maxsize=1)
def get_reference_tz():
    tz_name = tf.timezone_at(lat=referenceLatitude, lng=referenceLongitude)
    if tz_name is None:
        tz_name = tf.closest_timezone_at(lat=referenceLatitude, lng=referenceLongitude)
    if tz_name is None:
        tz_name = "Asia/Kolkata"
    return ZoneInfo(tz_name)

def get_reference_now():
    return datetime.now(get_reference_tz())

def create_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    # Keep images disabled for speed
    opts.add_argument("--blink-settings=imagesEnabled=false")
    if headless:
        # modern headless mode
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1280,800")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opts)

def routes_ready(driver, timeout=15):
    # wait for existence
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-trip-index]")))
    routes = driver.find_elements(By.CSS_SELECTOR, "div[data-trip-index]")
    if not routes:
        return False, []
    # Quick validation of the first card's text
    text = routes[0].text or ""
    looks_valid = ("km" in text) or ("min" in text)
    return looks_valid, routes

def get_maps_url(origin, destination):
    origin = urllib.parse.quote(origin)
    destination = urllib.parse.quote(destination)
    url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&travelmode=driving"
    return url

def get_route_points(route_code, format="short"):
    origin, destination = route_code.split("|")

    if format in ["long", "latlong"]:
        origin = olc.recoverNearest(origin, referenceLatitude, referenceLongitude)
        destination = olc.recoverNearest(destination, referenceLatitude, referenceLongitude)

    if format == "latlong":
        o_area = olc.decode(origin)
        d_area = olc.decode(destination)
        origin = f"{o_area.latitudeCenter},{o_area.longitudeCenter}"
        destination = f"{d_area.latitudeCenter},{d_area.longitudeCenter}"

    return origin, destination

def get_duration(s):
    # Handles: "25 min", "1 hr 5 min", "2 hr", "7 min"
    if not isinstance(s, str) or not s.strip():
        return np.nan
    parts = s.split()
    mins = 0
    try:
        if "hr" in parts:
            h_idx = parts.index("hr")
            mins += int(parts[h_idx - 1]) * 60
        if "min" in parts:
            m_idx = parts.index("min")
            mins += int(parts[m_idx - 1])
        # Fallback: if neither token present but a bare integer exists (rare)
        if "hr" not in parts and "min" not in parts:
            mins = float(parts[0])
    except Exception:
        return np.nan
    return mins

def get_traffic_report(driver, origin, destination, mode='car', max_retries=3, retry_delay=10):
    modes = {'bike': "\ue9f9", 'car': "\ue531", 'transit': "\ue535"}
    mode = modes.get(mode, modes['car'])

    attempts = 0
    while True:
        try:
            maps_url = get_maps_url(origin, destination)
            driver.get(maps_url)

            ok, routes = routes_ready(driver, timeout=15)
            if not ok:
                driver.refresh()
                ok, routes = routes_ready(driver, timeout=10)

            time_taken = None
            distance = None
            if routes:
                try:
                    parts0 = routes[0].text.split("\n")
                    time_taken = parts0[1] if len(parts0) > 1 else None
                    distance  = parts0[2] if len(parts0) > 2 else None
                except Exception:
                    pass
                for route in routes:
                    if mode in route.text:
                        parts = route.text.split("\n")
                        time_taken = parts[1] if len(parts) > 1 else time_taken
                        distance  = parts[2] if len(parts) > 2 else distance
                        break

            # Soft retry once if parse failed
            if time_taken is None or distance is None:
                driver.refresh()
                ok, routes = routes_ready(driver, timeout=10)
                if ok and routes:
                    try:
                        parts0 = routes[0].text.split("\n")
                        time_taken = parts0[1] if len(parts0) > 1 else time_taken
                        distance  = parts0[2] if len(parts0) > 2 else distance
                    except Exception:
                        pass

            return time_taken, distance

        except Exception:
            attempts += 1
            if attempts >= max_retries:
                raise
            time.sleep(retry_delay)

def main():
    driver = create_driver(headless=True)
    df = pd.DataFrame()
    now_ref = get_reference_now()
    date_now = now_ref.date()
    time_now = now_ref.strftime("%H:%M")
    try:
        for index, route in routes_df.iterrows():
            origin_pc, destination_pc = get_route_points(route["route_code"])
            origin_row = locations_df.loc[locations_df["plus_code"] == origin_pc, "location"]
            dest_row   = locations_df.loc[locations_df["plus_code"] == destination_pc, "location"]
            if origin_row.empty or dest_row.empty:
                continue

            origin = origin_row.iloc[0]
            destination = dest_row.iloc[0]

            travel_time, travel_distance = get_traffic_report(driver, origin, destination)

            new_row = {
                "date": date_now,
                "time": time_now,
                "route_code": route["route_code"],
                "duration": travel_time,
                "distance": travel_distance
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            time.sleep(0.3)
    finally:
        driver.quit()
    
    df["distance"] = pd.to_numeric(
        df["distance"].str.replace(" km", "", regex=False), errors="coerce")

    df["duration"] = df["duration"].apply(get_duration)

    df = df.dropna(subset=["distance", "duration"]).copy()

    df["distance"] = df["distance"].astype(float)
    df["duration"] = df["duration"].astype(int)

    raw_path = out_file + "_raw.csv"
    if os.path.exists(raw_path):
        df.to_csv(raw_path, mode="a", header=False, index=False)
    else:
        df.to_csv(raw_path, mode="w", header=df.columns, index=False)

    df_traffic = df.copy()
    df_traffic['avg_speed'] = round(df_traffic['distance'] / (df_traffic['duration'] / 60), 2)
    df_traffic['origin'] = df_traffic['route_code'].str.split('|').str[0]
    df_traffic['destination'] = df_traffic['route_code'].str.split('|').str[1]
    df_traffic = df_traffic.sort_values('avg_speed', ascending=True).reset_index(drop=True)
    df_traffic['origin'] = df_traffic['origin'].map(locations_df.set_index('plus_code')['location'])
    df_traffic['destination'] = df_traffic['destination'].map(locations_df.set_index('plus_code')['location'])
    df_traffic = df_traffic[['date', 'time', 'origin', 'destination', 'duration', 'distance', 'avg_speed']]

    processed_path = out_file + "_processed.csv"
    if os.path.exists(processed_path):
        df_traffic.to_csv(processed_path, mode="a", header=False, index=False)
    else:
        df_traffic.to_csv(processed_path, mode="w", header=df_traffic.columns, index=False)

    logs = df_traffic.tail(1)
    print(f"{logs['date'].iloc[0]} {logs['time'].iloc[0]} [traffic_snapshot] {logs['duration'].iloc[0]} mins from {logs['origin'].iloc[0]} to {logs['destination'].iloc[0]} - {logs['avg_speed'].iloc[0]} Km/hr.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())