"""
tiktok_lastfm_end_to_end_safe.py
-----------------------------------
Predict if a TikTok-trending song will go viral on SpotifyCharts
using Last.fm metadata (no rate limits).
"""

import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
import lightgbm as lgb
from io import StringIO
from tqdm import tqdm
from datetime import date, timedelta
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIGURATION
# -----------------------------
KAGGLE_FILE = "tiktok.csv"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

LASTFM_API_KEY = "YOUR_LASTFM_API_KEY"   # 🔑 <--- put your key here

PAST_YEARS = 3
LABEL_WINDOW_DAYS = 30
STREAMS_GROWTH_THRESHOLD = 3.0
MIN_TOTAL_STREAMS = 25_000_000
RANDOM_STATE = 42
REQUEST_DELAY = 0.3   # seconds between Last.fm requests

# -----------------------------
# 1. Load TikTok dataset
# -----------------------------
print("📥 Loading TikTok Trending Tracks dataset...")
kag = pd.read_csv(KAGGLE_FILE)
kag.columns = [c.lower() for c in kag.columns]

track_col = "track_name" if "track_name" in kag.columns else "track"
artist_col = "artist_name" if "artist_name" in kag.columns else "artist"
date_col = "release_date"
kag[date_col] = pd.to_datetime(kag[date_col], errors="coerce")

kag["combo"] = kag[track_col].str.lower().fillna('') + " - " + kag[artist_col].str.lower().fillna('')
unique_tracks = kag.drop_duplicates(subset=["combo"])[[track_col, artist_col, date_col]].dropna()
print(f"✅ Found {len(unique_tracks)} unique tracks.")

# -----------------------------
# 2. Fetch Last.fm metadata sequentially
# -----------------------------
print("🎧 Fetching Last.fm metadata sequentially (safe mode)...")
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

def fetch_lastfm_metadata(title, artist):
    """Fetch track + artist metadata from Last.fm sequentially."""
    params = {
        "method": "track.getInfo",
        "api_key": LASTFM_API_KEY,
        "artist": artist,
        "track": title,
        "format": "json"
    }

    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "track" not in data:
            return None

        track = data["track"]
        artist_name = track.get("artist", {}).get("name", artist)
        listeners = int(track.get("listeners", 0))
        playcount = int(track.get("playcount", 0))
        duration = int(track.get("duration", 0)) / 1000 if track.get("duration") else np.nan
        tags = [t["name"] for t in track.get("toptags", {}).get("tag", [])] if track.get("toptags") else []
        num_tags = len(tags)

        time.sleep(REQUEST_DELAY)
        return {
            "track_name": title,
            "artist_name": artist_name,
            "lastfm_listeners": listeners,
            "lastfm_playcount": playcount,
            "lastfm_duration_sec": duration,
            "lastfm_tags": ", ".join(tags),
            "lastfm_num_tags": num_tags
        }

    except Exception:
        time.sleep(REQUEST_DELAY)
        return None

lastfm_results = []
for _, row in tqdm(unique_tracks.iterrows(), total=len(unique_tracks), desc="Last.fm tracks"):
    meta = fetch_lastfm_metadata(row[track_col], row[artist_col])
    if meta:
        lastfm_results.append(meta)

lastfm_meta = pd.DataFrame(lastfm_results)
lastfm_meta.to_csv(os.path.join(DATA_DIR, "lastfm_metadata.csv"), index=False)
print(f"✅ Retrieved {len(lastfm_meta)} tracks with Last.fm metadata.")

# -----------------------------
# 3. SpotifyCharts data (public, safe)
# -----------------------------
print("📊 Downloading SpotifyCharts data sequentially...")

def fetch_chart_day(d, region="global"):
    url = f"https://spotifycharts.com/regional/{region}/daily/{d.isoformat()}/download"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text), skiprows=1)
            df["region"] = region
            df["date"] = d
            return df
    except:
        pass
    return None

start_date = date.today() - timedelta(days=PAST_YEARS * 365)
end_date = date.today()

dates = []
d = start_date
while d <= end_date:
    dates.append(d)
    d += timedelta(days=1)

all_data = []
for d in tqdm(dates, desc="SpotifyCharts"):
    df = fetch_chart_day(d)
    if df is not None:
        all_data.append(df)
    time.sleep(0.3)

if not all_data:
    print("❌ Failed to download chart data. Exiting.")
    exit(1)

spotify_charts = pd.concat(all_data, ignore_index=True)
spotify_charts.to_csv(os.path.join(DATA_DIR, "spotify_time_series.csv"), index=False)
print(f"✅ Downloaded {len(spotify_charts)} rows of SpotifyCharts data.")

# -----------------------------
# 4. Merge TikTok + Last.fm
# -----------------------------
merged = pd.merge(unique_tracks, lastfm_meta, on=["track_name", "artist_name"], how="left")
spotify_charts["spotify_id"] = spotify_charts["URL"].apply(lambda x: x.split("/")[-1] if isinstance(x, str) else None)
spotify_charts["date"] = pd.to_datetime(spotify_charts["date"], errors="coerce")

# -----------------------------
# 5. Viral Label
# -----------------------------
print("🏷️ Building viral label (3× growth + 25M streams)...")

labels = []
for _, row in tqdm(merged.iterrows(), total=len(merged)):
    tid = row.get("spotify_id")
    if pd.isna(tid):
        labels.append(np.nan)
        continue

    first_seen = pd.to_datetime(row[date_col])
    if pd.isna(first_seen):
        labels.append(np.nan)
        continue

    label_cutoff = first_seen + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    df = spotify_charts[
        (spotify_charts["spotify_id"] == tid) &
        (spotify_charts["date"] >= first_seen) &
        (spotify_charts["date"] <= label_cutoff)
    ].sort_values("date").copy()

    if len(df) < 2:
        labels.append(0)
        continue

    df["streams_shift7"] = df["Streams"].shift(7)
    df["pct_growth"] = (df["Streams"] - df["streams_shift7"]) / (df["streams_shift7"] + 1e-9)
    max_growth = df["pct_growth"].max()
    total_streams = df["Streams"].sum()

    viral = int((max_growth > STREAMS_GROWTH_THRESHOLD) and (total_streams >= MIN_TOTAL_STREAMS))
    labels.append(viral)

merged["viral_label"] = labels
merged.to_csv(os.path.join(DATA_DIR, "merged_tiktok_lastfm.csv"), index=False)
print(f"✅ Created viral_label with {merged['viral_label'].sum()} viral songs out of {len(merged)} total.")

# -----------------------------
# 6. Train LightGBM
# -----------------------------
print("🎯 Training LightGBM classifier...")

train_df = merged.dropna(subset=["viral_label"])
train_df["viral_label"] = train_df["viral_label"].astype(int)

feature_cols = [
    "lastfm_listeners",
    "lastfm_playcount",
    "lastfm_duration_sec",
    "lastfm_num_tags"
]

feature_cols = [f for f in feature_cols if f in train_df.columns]
X = train_df[feature_cols].fillna(0)
y = train_df["viral_label"]

if len(y.unique()) < 2:
    print("⚠️ Not enough samples to train model.")
else:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "is_unbalance": True,
        "seed": RANDOM_STATE
    }

    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    lgb_train = lgb.Dataset(train_x, label=train_y)
    lgb_val = lgb.Dataset(val_x, label=val_y)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict(val_x, num_iteration=model.best_iteration)
    auc = roc_auc_score(val_y, preds)
    pr = average_precision_score(val_y, preds)

    print(f"✅ Model Performance: AUC={auc:.4f}, PR-AUC={pr:.4f}")
    joblib.dump(model, os.path.join(DATA_DIR, "lgb_tiktok_lastfm.pkl"))
    print("💾 Model saved to ./data/lgb_tiktok_lastfm.pkl")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance()
    }).sort_values("importance", ascending=False)

    print("\n🏆 Top predictive features:")
    print(importance)

print("🏁 Done! Sequential Last.fm version complete.")
