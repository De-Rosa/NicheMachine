"""
tiktok_full_pipeline_parallel_mbid_cache.py (FIXED)
-----------------------------------
Predict if a TikTok-trending song will go viral using Last.fm, AcousticBrainz, and SpotifyCharts.
Includes automatic MBID lookup via MusicBrainz (parallelized) with caching.

FIXES:
1. Fixed column name consistency in merges
2. Fixed MBID fetching to use correct column names
3. Fixed AcousticBrainz feature extraction
4. Added rate limiting for API calls
5. Replaced Spotify-based viral detection with Last.fm metrics
6. Added proper error handling
7. Fixed merge operations with consistent column names
8. Added engagement ratio (listeners/plays) as viral indicator
9. Added recency bonus for newer tracks
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
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import musicbrainzngs

# -----------------------------
# CONFIGURATION
# -----------------------------
KAGGLE_FILE = "tiktok.csv"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

LASTFM_API_KEY = "API_KEY"
PAST_YEARS = 3
LABEL_WINDOW_DAYS = 30

# Last.fm-based thresholds (much lower than Spotify streams)
PLAYCOUNT_GROWTH_THRESHOLD = 2.5  # 250% growth in playcount
MIN_TOTAL_PLAYCOUNT = 1_000_000  # 1M total plays
MIN_LISTENERS = 50_000  # 50k unique listeners
LISTENER_TO_PLAY_RATIO_MIN = 0.02  # At least 2% listener/play ratio (engagement)

RANDOM_STATE = 42

MAX_WORKERS_API = 5  # Reduced to avoid rate limiting
REQUEST_TIMEOUT = 10
RATE_LIMIT_DELAY = 0.1  # Delay between requests

# Initialize MusicBrainz
musicbrainzngs.set_useragent("TikTokViralPredictor", "1.0", "your-email@example.com")

# Thread-safe rate limiter
rate_limit_lock = Lock()
last_request_time = 0


def rate_limited_sleep():
    """Simple rate limiter"""
    global last_request_time
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last)
        last_request_time = time.time()


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

# Standardize column names to avoid merge issues
kag = kag.rename(columns={track_col: "track_name", artist_col: "artist_name"})

kag["combo"] = kag["track_name"].str.lower().fillna('') + " - " + kag["artist_name"].str.lower().fillna('')
unique_tracks = kag.drop_duplicates(subset=["combo"])[["track_name", "artist_name", date_col]].dropna()
print(f"✅ Found {len(unique_tracks)} unique tracks.")

# -----------------------------
# 2. Fetch Last.fm metadata (parallel)
# -----------------------------
print("🔑 Fetching Last.fm metadata...")
lastfm_results = []
lastfm_lock = Lock()


def fetch_lastfm_track(row):
    title, artist = row["track_name"], row["artist_name"]
    rate_limited_sleep()
    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist)}&track={requests.utils.quote(title)}&format=json"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        track_info = data.get("track", {})
        result = {
            "track_name": title,
            "artist_name": artist,
            "lastfm_listeners": int(track_info.get("listeners", 0)),
            "lastfm_playcount": int(track_info.get("playcount", 0)),
            "lastfm_url": track_info.get("url", "")
        }
        with lastfm_lock:
            lastfm_results.append(result)
        return result
    except Exception as e:
        return None


with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as executor:
    list(tqdm(executor.map(fetch_lastfm_track, [row for _, row in unique_tracks.iterrows()]),
              total=len(unique_tracks), desc="Fetching Last.fm tracks"))

lastfm_meta = pd.DataFrame(lastfm_results)
if len(lastfm_meta) > 0:
    lastfm_meta.to_csv(os.path.join(DATA_DIR, "lastfm_metadata.csv"), index=False)
print(f"✅ Retrieved {len(lastfm_meta)} Last.fm tracks.")

# -----------------------------
# 3. Fetch MBIDs from MusicBrainz (parallelized with caching)
# -----------------------------
print("🎯 Fetching MusicBrainz MBIDs with caching...")

MBID_CACHE_FILE = os.path.join(DATA_DIR, "mbids.csv")

# Load cache if exists
if os.path.exists(MBID_CACHE_FILE):
    mbid_meta = pd.read_csv(MBID_CACHE_FILE)
    cached_combos = set(mbid_meta["track_name"].str.lower() + " - " + mbid_meta["artist_name"].str.lower())
    print(f"✅ Loaded {len(mbid_meta)} cached MBIDs.")
else:
    mbid_meta = pd.DataFrame(columns=["track_name", "artist_name", "mbid"])
    cached_combos = set()

mbid_results = []
mbid_lock = Lock()
mbid_save_counter = 0
SAVE_INTERVAL = 50  # Save cache every 50 new MBIDs


def fetch_mbid(row):
    global mbid_save_counter, mbid_meta
    title, artist = row["track_name"], row["artist_name"]
    combo = f"{title.lower()} - {artist.lower()}"
    if combo in cached_combos:
        return

    rate_limited_sleep()
    try:
        result = musicbrainzngs.search_recordings(recording=title, artist=artist, limit=1)
        recordings = result.get("recording-list", [])
        mbid = recordings[0]["id"] if recordings else None
    except Exception as e:
        mbid = None

    with mbid_lock:
        mbid_results.append({"track_name": title, "artist_name": artist, "mbid": mbid})
        mbid_save_counter += 1

        # Incremental save every SAVE_INTERVAL tracks
        if mbid_save_counter >= SAVE_INTERVAL:
            temp_meta = pd.concat([mbid_meta, pd.DataFrame(mbid_results)], ignore_index=True)
            temp_meta.to_csv(MBID_CACHE_FILE, index=False)
            mbid_meta = temp_meta
            print(f"\n💾 Auto-saved {len(mbid_results)} MBIDs to cache (total: {len(mbid_meta)})")
            mbid_results.clear()
            mbid_save_counter = 0


tracks_to_fetch = [row for _, row in unique_tracks.iterrows()
                   if f"{row['track_name'].lower()} - {row['artist_name'].lower()}" not in cached_combos]

if tracks_to_fetch:
    print(f"🔍 Need to fetch {len(tracks_to_fetch)} new MBIDs...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as executor:
        list(tqdm(executor.map(fetch_mbid, tracks_to_fetch), total=len(tracks_to_fetch),
                  desc="Fetching MBIDs"))

    # Final save for any remaining results
    if mbid_results:
        mbid_meta = pd.concat([mbid_meta, pd.DataFrame(mbid_results)], ignore_index=True)
        mbid_meta.to_csv(MBID_CACHE_FILE, index=False)
        print(f"✅ Final save: Updated MBID cache with {len(mbid_results)} new entries.")

print(f"✅ Total MBIDs available: {mbid_meta['mbid'].notna().sum()}")

# -----------------------------
# 4. Fetch AcousticBrainz features (parallel)
# -----------------------------
print("🎵 Fetching AcousticBrainz features...")

acoustic_results = []
acoustic_lock = Lock()


def fetch_acoustic_features(row):
    mbid = row.get("mbid")
    title, artist = row["track_name"], row["artist_name"]

    if pd.isna(mbid):
        result = {"track_name": title, "artist_name": artist,
                  "acoustic_danceability": None, "acoustic_tempo": None,
                  "acoustic_energy": None, "acoustic_valence": None}
    else:
        rate_limited_sleep()
        try:
            url = f"https://acousticbrainz.org/api/v1/{mbid}/high-level"
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                result = {"track_name": title, "artist_name": artist,
                          "acoustic_danceability": None, "acoustic_tempo": None,
                          "acoustic_energy": None, "acoustic_valence": None}
            else:
                data = r.json()
                highlevel = data.get("highlevel", {})
                rhythm = data.get("rhythm", {})

                # Extract values properly
                danceability = rhythm.get("danceability", {})
                danceability_val = danceability.get("value") if isinstance(danceability, dict) else danceability

                tempo = rhythm.get("bpm")

                energy = highlevel.get("mood_acoustic", {})
                energy_val = 1.0 - energy.get("probability", 0) if isinstance(energy, dict) else None

                valence = highlevel.get("mood_happy", {})
                valence_val = valence.get("probability") if isinstance(valence, dict) else valence

                result = {
                    "track_name": title,
                    "artist_name": artist,
                    "acoustic_danceability": danceability_val,
                    "acoustic_tempo": tempo,
                    "acoustic_energy": energy_val,
                    "acoustic_valence": valence_val
                }
        except Exception as e:
            result = {"track_name": title, "artist_name": artist,
                      "acoustic_danceability": None, "acoustic_tempo": None,
                      "acoustic_energy": None, "acoustic_valence": None}

    with acoustic_lock:
        acoustic_results.append(result)


with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as executor:
    list(tqdm(executor.map(fetch_acoustic_features, [row for _, row in mbid_meta.iterrows()]),
              total=len(mbid_meta), desc="Fetching AcousticBrainz features"))

acoustic_meta = pd.DataFrame(acoustic_results)
if len(acoustic_meta) > 0:
    acoustic_meta.to_csv(os.path.join(DATA_DIR, "acousticbrainz_features.csv"), index=False)
print(f"✅ Retrieved {len(acoustic_meta)} AcousticBrainz tracks.")

# -----------------------------
# 5. Download SpotifyCharts global data
# -----------------------------
print("📊 Downloading SpotifyCharts global data...")


def fetch_global_chart_day(d):
    rate_limited_sleep()
    url = f"https://spotifycharts.com/regional/global/daily/{d.isoformat()}/download"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200 or "html" in r.text.lower()[:100]:
            return None
        df = pd.read_csv(StringIO(r.text), skiprows=1)
        df["region"] = "global"
        df["date"] = d
        return df
    except Exception as e:
        return None


all_chart_data = []
start_date = date.today() - timedelta(days=PAST_YEARS * 365)
end_date = date.today()
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

for d in tqdm(dates, desc="Downloading global charts"):
    df = fetch_global_chart_day(d)
    if df is not None:
        all_chart_data.append(df)

if not all_chart_data:
    print("❌ Failed to download any chart data. Creating dummy data for testing...")
    # Create minimal dummy data to allow pipeline to complete
    spotify_charts = pd.DataFrame({
        "track name": [], "artist": [], "streams": [],
        "url": [], "date": [], "region": []
    })
else:
    spotify_charts = pd.concat(all_chart_data, ignore_index=True)

spotify_charts.columns = [c.strip().lower() for c in spotify_charts.columns]

track_col_chart = next((c for c in spotify_charts.columns if "track" in c), "track name")
artist_col_chart = next((c for c in spotify_charts.columns if "artist" in c), "artist")
streams_col = next((c for c in spotify_charts.columns if "streams" in c), "streams")
url_col = next((c for c in spotify_charts.columns if "url" in c or "uri" in c), "url")

if len(spotify_charts) > 0:
    spotify_charts["spotify_id"] = spotify_charts[url_col].apply(
        lambda x: str(x).split("/")[-1] if pd.notna(x) else None
    )
    spotify_charts["date"] = pd.to_datetime(spotify_charts["date"], errors="coerce")
    spotify_charts.to_csv(os.path.join(DATA_DIR, "spotify_time_series.csv"), index=False)
    print(f"✅ Downloaded {len(spotify_charts)} rows of SpotifyCharts data.")

# -----------------------------
# 6. Merge all datasets
# -----------------------------
print("🔄 Merging datasets...")
merged = unique_tracks.copy()

if len(lastfm_meta) > 0:
    merged = merged.merge(lastfm_meta, on=["track_name", "artist_name"], how="left")
if len(mbid_meta) > 0:
    merged = merged.merge(mbid_meta, on=["track_name", "artist_name"], how="left")
if len(acoustic_meta) > 0:
    merged = merged.merge(acoustic_meta, on=["track_name", "artist_name"], how="left")

merged.to_csv(os.path.join(DATA_DIR, "merged_full.csv"), index=False)
print(f"✅ Merged dataset has {len(merged)} tracks.")

# -----------------------------
# 7. Build viral labels using Last.fm data
# -----------------------------
print("🏷️ Building viral labels based on Last.fm metrics...")


def calculate_viral_label(row):
    """
    Determine if a track went viral based on Last.fm metrics:
    - High playcount (total plays)
    - High listener count (unique users)
    - Good engagement ratio (listeners/plays)
    - Recency bonus for newer tracks
    """
    playcount = row.get("lastfm_playcount", 0)
    listeners = row.get("lastfm_listeners", 0)
    release_date = pd.to_datetime(row.get(date_col))

    # Convert to numeric if needed
    try:
        playcount = float(playcount) if pd.notna(playcount) else 0
        listeners = float(listeners) if pd.notna(listeners) else 0
    except:
        playcount = 0
        listeners = 0

    # Basic thresholds
    if playcount < MIN_TOTAL_PLAYCOUNT or listeners < MIN_LISTENERS:
        return 0

    # Engagement ratio (listeners/playcount)
    engagement_ratio = listeners / (playcount + 1e-9)
    if engagement_ratio < LISTENER_TO_PLAY_RATIO_MIN:
        return 0  # Low engagement = not viral

    # Recency bonus: tracks released in last 2 years get lower thresholds
    if pd.notna(release_date):
        days_since_release = (pd.Timestamp.now() - release_date).days
        is_recent = days_since_release < 730  # Within 2 years

        if is_recent:
            # More lenient for recent tracks
            viral_threshold_playcount = MIN_TOTAL_PLAYCOUNT * 0.7
            viral_threshold_listeners = MIN_LISTENERS * 0.7
        else:
            # Stricter for older tracks
            viral_threshold_playcount = MIN_TOTAL_PLAYCOUNT * 1.5
            viral_threshold_listeners = MIN_LISTENERS * 1.5

        if playcount >= viral_threshold_playcount and listeners >= viral_threshold_listeners:
            return 1
    else:
        # No release date, use standard thresholds
        if playcount >= MIN_TOTAL_PLAYCOUNT and listeners >= MIN_LISTENERS:
            return 1

    return 0


labels = []
for _, row in tqdm(merged.iterrows(), total=len(merged)):
    viral = calculate_viral_label(row)
    labels.append(viral)

merged["viral_label"] = labels

# Add additional metrics for analysis
merged["listener_play_ratio"] = merged["lastfm_listeners"] / (merged["lastfm_playcount"] + 1e-9)
merged["days_since_release"] = (pd.Timestamp.now() - pd.to_datetime(merged[date_col])).dt.days

merged.to_csv(os.path.join(DATA_DIR, "merged_labeled.csv"), index=False)
print(f"✅ Viral labels assigned: {merged['viral_label'].sum()} viral tracks out of {len(merged)}.")
print(f"   Viral rate: {100 * merged['viral_label'].sum() / len(merged):.1f}%")

# -----------------------------
# 8. Train LightGBM classifier
# -----------------------------
print("🎯 Training LightGBM classifier...")

train_df = merged.dropna(subset=["viral_label"]).copy()
train_df["viral_label"] = train_df["viral_label"].astype(int)

feature_cols = [
    "lastfm_listeners", "lastfm_playcount",
    "listener_play_ratio",  # Engagement metric
    "acoustic_danceability", "acoustic_tempo",
    "acoustic_energy", "acoustic_valence"
]
feature_cols = [f for f in feature_cols if f in train_df.columns]

X = train_df[feature_cols].fillna(0)
y = train_df["viral_label"]

if len(y.unique()) < 2:
    print("⚠️ Not enough positive/negative samples to train model.")
    print(f"   Found {y.sum()} positive samples and {len(y) - y.sum()} negative samples.")
else:
    print(f"Training with {len(X)} samples ({y.sum()} positive, {len(y) - y.sum()} negative)")
    lgb_train = lgb.Dataset(X, label=y)
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
        "seed": RANDOM_STATE,
        "device": "cpu"
    }
    model = lgb.train(params, lgb_train, num_boost_round=500)
    joblib.dump(model, os.path.join(DATA_DIR, "lgb_full_pipeline.pkl"))
    print("✅ LightGBM model trained and saved.")


print("🏁 Full pipeline complete!")
