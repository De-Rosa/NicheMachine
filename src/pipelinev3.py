"""
tiktok_pipeline_breakthrough_viral.py
-----------------------------------
Complete pipeline with REDEFINED viral labels based on breakthrough metrics:
- Early catch-on vs artist baseline
- Outperformance (not absolute numbers)
- Viral velocity (engagement ratio)
"""

import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
import musicbrainzngs

# -----------------------------
# CONFIGURATION
# -----------------------------
KAGGLE_FILE = "tiktok.csv"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

LASTFM_API_KEY = "API_KEY"
RANDOM_STATE = 42

# NEW VIRAL CRITERIA (Breakthrough-based)
VIRAL_CRITERIA = {
    "early_engagement_multiplier": 1.5,  # 1.5x artist's typical early engagement
    "artist_outperformance_multiplier": 2.0,  # 2x artist's average listeners
    "viral_engagement_threshold": 0.03,  # 3% listener/play ratio
    "min_listeners": 10_000,  # Minimum 10k listeners
}

MAX_WORKERS_API = 5
REQUEST_TIMEOUT = 10
RATE_LIMIT_DELAY = 0.1

musicbrainzngs.set_useragent("TikTokViralPredictor", "1.0", "your-email@example.com")

rate_limit_lock = Lock()
last_request_time = 0


def rate_limited_sleep():
    global last_request_time
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last)
        last_request_time = time.time()


print("🎯 TikTok Viral Prediction - Breakthrough Detection")
print("=" * 70)
print("\n💡 NEW Viral Definition:")
print("   A track is viral if it meets 2+ criteria:")
print(f"   1. Early Engagement: {VIRAL_CRITERIA['early_engagement_multiplier']}x artist's typical engagement")
print(f"   2. Outperformance: {VIRAL_CRITERIA['artist_outperformance_multiplier']}x artist's average")
print(f"   3. High Velocity: >{VIRAL_CRITERIA['viral_engagement_threshold']:.1%} engagement ratio")
print("=" * 70)

# -----------------------------
# 1. Load TikTok dataset
# -----------------------------
print("\n📥 Loading TikTok Trending Tracks dataset...")
kag = pd.read_csv(KAGGLE_FILE)
kag.columns = [c.lower() for c in kag.columns]

track_col = "track_name" if "track_name" in kag.columns else "track"
artist_col = "artist_name" if "artist_name" in kag.columns else "artist"
date_col = "release_date"
kag[date_col] = pd.to_datetime(kag[date_col], errors="coerce")

kag = kag.rename(columns={track_col: "track_name", artist_col: "artist_name"})

kag["combo"] = kag["track_name"].str.lower().fillna('') + " - " + kag["artist_name"].str.lower().fillna('')
unique_tracks = kag.drop_duplicates(subset=["combo"])[["track_name", "artist_name", date_col]].dropna()
print(f"✅ Found {len(unique_tracks)} unique tracks.")

# -----------------------------
# 2. Fetch Last.fm metadata (parallel)
# -----------------------------
print("\n🔑 Fetching Last.fm metadata...")
LASTFM_CACHE = os.path.join(DATA_DIR, "lastfm_metadata.csv")

if os.path.exists(LASTFM_CACHE):
    lastfm_meta = pd.read_csv(LASTFM_CACHE)
    cached_lastfm = set(lastfm_meta["track_name"] + "|" + lastfm_meta["artist_name"])
    print(f"✅ Loaded {len(lastfm_meta)} cached Last.fm tracks")
else:
    lastfm_meta = pd.DataFrame()
    cached_lastfm = set()

lastfm_results = []
lastfm_lock = Lock()


def fetch_lastfm_track(row):
    title, artist = row["track_name"], row["artist_name"]

    cache_key = f"{title}|{artist}"
    if cache_key in cached_lastfm:
        return

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
    except:
        return None


tracks_to_fetch = [row for _, row in unique_tracks.iterrows()
                   if f"{row['track_name']}|{row['artist_name']}" not in cached_lastfm]

if tracks_to_fetch:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as executor:
        list(tqdm(executor.map(fetch_lastfm_track, tracks_to_fetch),
                  total=len(tracks_to_fetch), desc="Fetching Last.fm tracks"))

    if lastfm_results:
        lastfm_meta = pd.concat([lastfm_meta, pd.DataFrame(lastfm_results)], ignore_index=True)
        lastfm_meta.to_csv(LASTFM_CACHE, index=False)

print(f"✅ Total Last.fm tracks: {len(lastfm_meta)}")

# -----------------------------
# 3. FEATURE ENGINEERING: Fetch Last.fm Tags
# -----------------------------
print("\n🏷️  FEATURE ENGINEERING: Fetching Last.fm tags...")

TAGS_CACHE = os.path.join(DATA_DIR, "lastfm_tags.csv")

if os.path.exists(TAGS_CACHE):
    tags_meta = pd.read_csv(TAGS_CACHE)
    cached_tags = set(tags_meta["track_name"] + "|" + tags_meta["artist_name"])
    print(f"✅ Loaded {len(tags_meta)} cached tag records")
else:
    tags_meta = pd.DataFrame()
    cached_tags = set()

tags_results = []
tags_lock = Lock()


def fetch_track_tags(row):
    """Fetch Last.fm tags and compute acoustic-like features"""
    title, artist = row["track_name"], row["artist_name"]

    cache_key = f"{title}|{artist}"
    if cache_key in cached_tags:
        return

    rate_limited_sleep()
    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=track.getTopTags&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist)}&track={requests.utils.quote(title)}&format=json"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None

        data = r.json()
        tags = data.get("toptags", {}).get("tag", [])

        if not tags:
            with tags_lock:
                tags_results.append({
                    "track_name": title,
                    "artist_name": artist,
                    "raw_tags": "",
                    "num_tags": 0
                })
            return

        tag_dict = {}
        for t in tags[:30]:
            tag_name = t.get("name", "").lower()
            tag_count = int(t.get("count", 0))
            if tag_name:
                tag_dict[tag_name] = tag_count

        result = {
            "track_name": title,
            "artist_name": artist,
            "raw_tags": ",".join(list(tag_dict.keys())[:20]),
            "num_tags": len(tag_dict)
        }

        with tags_lock:
            tags_results.append(result)

        return result

    except:
        return None


tracks_to_tag = [row for _, row in unique_tracks.iterrows()
                 if f"{row['track_name']}|{row['artist_name']}" not in cached_tags]

if tracks_to_tag:
    print(f"Fetching tags for {len(tracks_to_tag)} tracks...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as executor:
        list(tqdm(executor.map(fetch_track_tags, tracks_to_tag),
                  total=len(tracks_to_tag), desc="Fetching tags"))

    if tags_results:
        tags_meta = pd.concat([tags_meta, pd.DataFrame(tags_results)], ignore_index=True)
        tags_meta.to_csv(TAGS_CACHE, index=False)

print(f"✅ Total tracks with tags: {len(tags_meta)}")

# -----------------------------
# 4. ADVANCED FEATURE ENGINEERING: Tag-to-Feature Mapping
# -----------------------------
print("\n🧪 ADVANCED FEATURE ENGINEERING: Computing acoustic features from tags...")


def engineer_acoustic_features(raw_tags_str):
    if pd.isna(raw_tags_str) or not raw_tags_str:
        return {
            "fe_danceability": 0.5,
            "fe_energy": 0.5,
            "fe_valence": 0.5,
            "fe_tempo": 120,
            "fe_acousticness": 0.5,
            "fe_has_tags": 0
        }

    tags = set(raw_tags_str.lower().split(","))

    # DANCEABILITY
    dance_indicators = {
        "dance", "electronic", "edm", "house", "techno", "disco", "funk",
        "dancehall", "electro", "trance", "dubstep", "drum and bass",
        "dance-pop", "electropop", "synthpop", "dance-punk"
    }
    danceability_score = len(tags & dance_indicators) / 5.0
    danceability = min(0.95, max(0.2, 0.4 + danceability_score * 0.5))

    # ENERGY
    high_energy_tags = {
        "energetic", "upbeat", "party", "hype", "aggressive", "intense",
        "powerful", "loud", "hard rock", "metal", "punk", "hardcore",
        "fast", "driving", "anthemic"
    }
    low_energy_tags = {
        "chill", "relaxing", "calm", "ambient", "mellow", "smooth",
        "peaceful", "tranquil", "downtempo", "lounge", "easy listening"
    }
    energy_boost = len(tags & high_energy_tags) / 5.0
    energy_penalty = len(tags & low_energy_tags) / 5.0
    energy = min(0.95, max(0.1, 0.5 + energy_boost * 0.4 - energy_penalty * 0.3))

    # VALENCE
    positive_tags = {
        "happy", "feel-good", "uplifting", "cheerful", "fun", "positive",
        "joyful", "optimistic", "bright", "summer", "sunny"
    }
    negative_tags = {
        "sad", "melancholy", "emotional", "depressing", "dark", "melancholic",
        "gloomy", "somber", "tragic", "lonely", "heartbreak", "emo"
    }
    valence_boost = len(tags & positive_tags) / 4.0
    valence_penalty = len(tags & negative_tags) / 4.0
    valence = min(0.95, max(0.1, 0.5 + valence_boost * 0.35 - valence_penalty * 0.35))

    # TEMPO
    fast_genres = {
        "punk", "metal", "drum and bass", "hardcore", "speed metal",
        "thrash metal", "power metal", "happy hardcore", "gabber"
    }
    slow_genres = {
        "ambient", "drone", "downtempo", "trip-hop", "ballad",
        "slow", "blues", "jazz", "lounge"
    }
    medium_fast = {"rock", "pop", "indie", "alternative"}

    if tags & fast_genres:
        tempo = 165
    elif tags & slow_genres:
        tempo = 85
    elif tags & medium_fast:
        tempo = 125
    else:
        tempo = 120

    # ACOUSTICNESS
    acoustic_tags = {
        "acoustic", "folk", "singer-songwriter", "classical", "jazz",
        "blues", "country", "unplugged", "stripped"
    }
    electronic_tags = {
        "electronic", "edm", "techno", "house", "electro", "synth",
        "industrial", "glitch"
    }
    acoustic_score = len(tags & acoustic_tags) / 3.0
    electronic_score = len(tags & electronic_tags) / 5.0
    acousticness = min(0.95, max(0.05, 0.5 + acoustic_score * 0.4 - electronic_score * 0.3))

    return {
        "fe_danceability": round(danceability, 3),
        "fe_energy": round(energy, 3),
        "fe_valence": round(valence, 3),
        "fe_tempo": int(tempo),
        "fe_acousticness": round(acousticness, 3),
        "fe_has_tags": 1
    }


engineered_features = tags_meta["raw_tags"].apply(engineer_acoustic_features)
engineered_df = pd.DataFrame(engineered_features.tolist())
tags_meta = pd.concat([tags_meta, engineered_df], axis=1)

tags_meta.to_csv(os.path.join(DATA_DIR, "lastfm_tags_engineered.csv"), index=False)
print(f"✅ Engineered {len(engineered_df.columns)} acoustic-like features from tags")

# -----------------------------
# 5. Merge all data
# -----------------------------
print("\n🔄 Merging all datasets...")

merged = unique_tracks.copy()

if len(lastfm_meta) > 0:
    merged = merged.merge(lastfm_meta, on=["track_name", "artist_name"], how="left")
if len(tags_meta) > 0:
    merged = merged.merge(tags_meta, on=["track_name", "artist_name"], how="left")

merged.to_csv(os.path.join(DATA_DIR, "merged_full_engineered.csv"), index=False)
print(f"✅ Merged dataset has {len(merged)} tracks")

# -----------------------------
# 6. Calculate artist baselines for BREAKTHROUGH detection
# -----------------------------
print("\n📈 Computing artist baselines for breakthrough detection...")

artist_stats = merged.groupby("artist_name").agg({
    "lastfm_listeners": ["mean", "median", "std", "count"],
    "lastfm_playcount": ["mean", "median"]
}).reset_index()

artist_stats.columns = [
    "artist_name", "artist_mean_listeners", "artist_median_listeners",
    "artist_std_listeners", "artist_track_count",
    "artist_mean_playcount", "artist_median_playcount"
]

merged = merged.merge(artist_stats, on="artist_name", how="left")
print(f"✅ Baselines computed for {len(artist_stats)} artists")

# -----------------------------
# 7. NEW VIRAL LABELING: Breakthrough-based
# -----------------------------
print("\n🏷️  Applying BREAKTHROUGH-based viral labels...")

# Compute breakthrough metrics
merged["listener_play_ratio"] = merged["lastfm_listeners"] / (merged["lastfm_playcount"] + 1e-9)
merged["listeners_vs_artist_mean"] = merged["lastfm_listeners"] / (merged["artist_mean_listeners"] + 1)

# Early engagement (viral tracks catch on faster)
merged["early_listeners"] = merged["lastfm_listeners"] * 0.05
merged["early_playcount"] = merged["lastfm_playcount"] * 0.03
merged["early_engagement_ratio"] = merged["early_listeners"] / (merged["early_playcount"] + 1e-9)
merged["artist_mean_engagement"] = merged["artist_mean_listeners"] / (merged["artist_mean_playcount"] + 1e-9)
merged["early_engagement_vs_artist"] = merged["early_engagement_ratio"] / (merged["artist_mean_engagement"] + 1e-9)


def calculate_viral_breakthrough(row):
    """Determine if track is a viral breakthrough"""
    listeners = row.get("lastfm_listeners", 0)

    if pd.isna(listeners) or listeners < VIRAL_CRITERIA["min_listeners"]:
        return 0

    criteria_met = 0

    # Criterion 1: Early engagement outperformance
    if row.get("early_engagement_vs_artist", 0) >= VIRAL_CRITERIA["early_engagement_multiplier"]:
        criteria_met += 1

    # Criterion 2: Outperforms artist baseline
    if row.get("listeners_vs_artist_mean", 0) >= VIRAL_CRITERIA["artist_outperformance_multiplier"]:
        criteria_met += 1

    # Criterion 3: High viral velocity
    if row.get("listener_play_ratio", 0) >= VIRAL_CRITERIA["viral_engagement_threshold"]:
        criteria_met += 1

    return 1 if criteria_met >= 2 else 0


merged["viral_label"] = merged.apply(calculate_viral_breakthrough, axis=1)

viral_count = merged["viral_label"].sum()
print(f"✅ Breakthrough labels assigned: {viral_count} viral ({100 * viral_count / len(merged):.1f}%)")
print(f"   Non-viral: {len(merged) - viral_count} ({100 * (len(merged) - viral_count) / len(merged):.1f}%)")

# Additional features
merged["log_early_listeners"] = np.log1p(merged["early_listeners"].fillna(0))
merged["log_early_playcount"] = np.log1p(merged["early_playcount"].fillna(0))
merged["log_artist_baseline"] = np.log1p(merged["artist_mean_listeners"].fillna(0))
merged["early_velocity"] = merged["early_playcount"] / (merged["early_listeners"] + 1)
merged["days_since_release"] = (pd.Timestamp.now() - pd.to_datetime(merged[date_col])).dt.days
merged["is_recent"] = (merged["days_since_release"] < 730).astype(int)

# Feature interactions
merged["dance_x_energy"] = merged["fe_danceability"] * merged["fe_energy"]
merged["party_score"] = (merged["fe_energy"] + merged["fe_valence"]) / 2

merged.to_csv(os.path.join(DATA_DIR, "merged_breakthrough_labeled.csv"), index=False)

# Show sample breakthroughs
print("\n🔥 Sample Viral Breakthroughs:")
viral_tracks = merged[merged["viral_label"] == 1].sort_values("listeners_vs_artist_mean", ascending=False)
if len(viral_tracks) > 0:
    for i, (_, row) in enumerate(viral_tracks.head(3).iterrows(), 1):
        print(f"   {i}. {row['track_name'][:35]:35s} - {row['artist_name'][:20]:20s}")
        print(f"      {row['listeners_vs_artist_mean']:.1f}x artist avg | Engagement: {row['listener_play_ratio']:.3f}")

# -----------------------------
# 8. Train model with BREAKTHROUGH labels
# -----------------------------
print("\n🎯 Training model with breakthrough-based labels...")

train_df = merged.dropna(subset=["viral_label"]).copy()
train_df["viral_label"] = train_df["viral_label"].astype(int)

# Feature set optimized for breakthrough detection
feature_cols = [
    # Early signals (most important for breakthrough!)
    "log_early_listeners",
    "log_early_playcount",
    "early_velocity",
    "early_engagement_ratio",
    "early_engagement_vs_artist",  # NEW: vs artist baseline

    # Artist context (critical for breakthrough!)
    "log_artist_baseline",
    "listeners_vs_artist_mean",  # NEW: outperformance metric

    # Engagement
    "listener_play_ratio",

    # Audio features (from tags)
    "fe_danceability",
    "fe_energy",
    "fe_valence",
    "fe_tempo",
    "fe_acousticness",

    # Interactions
    "dance_x_energy",
    "party_score",

    # Metadata
    "is_recent",
    "fe_has_tags"
]

feature_cols = [f for f in feature_cols if f in train_df.columns]

X = train_df[feature_cols].fillna(0)
y = train_df["viral_label"]

print(f"\n📊 Training Data:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(feature_cols)}")
print(f"   Viral breakthroughs: {y.sum()} ({100 * y.sum() / len(y):.1f}%)")
print(f"   Non-viral: {len(y) - y.sum()} ({100 * (len(y) - y.sum()) / len(y):.1f}%)")

if len(y.unique()) < 2:
    print("\n⚠️ Not enough class diversity. Adjust VIRAL_CRITERIA thresholds.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "is_unbalance": True,
    "seed": RANDOM_STATE
}

model = lgb.train(
    params, lgb_train, num_boost_round=500,
    valid_sets=[lgb_train, lgb_test],
    valid_names=["train", "test"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Evaluate
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

print(f"\n✅ Model Performance (Breakthrough Detection):")
print(f"   ROC-AUC: {auc:.4f}")
print(f"   Average Precision: {ap:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Not Viral', 'Viral Breakthrough'])}")

# Feature importance
print("\n🎯 Top Features for Detecting Breakthrough:")
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for _, row in feature_imp.head(10).iterrows():
    print(f"   {row['feature']:35s}: {row['importance']:>8,.0f}")

# Save
joblib.dump(model, os.path.join(DATA_DIR, "lgb_breakthrough_detector.pkl"))
with open(os.path.join(DATA_DIR, "features_breakthrough.txt"), 'w') as f:
    f.write('\n'.join(feature_cols))

print(f"\n💾 Model saved: {DATA_DIR}/lgb_breakthrough_detector.pkl")
print(f"💾 Features saved: {DATA_DIR}/features_breakthrough.txt")

print("\n" + "=" * 70)
print("🏁 Pipeline Complete!")
print("\n💡 This model detects BREAKTHROUGH viral success:")
print("   ✅ Small artist doing 10x their average = VIRAL")
print("   ❌ Big artist doing their typical numbers = NOT VIRAL")
print("\n📊 Key Metrics Used:")
print(f"   • Early engagement vs artist baseline ({VIRAL_CRITERIA['early_engagement_multiplier']}x)")
print(f"   • Outperformance vs artist average ({VIRAL_CRITERIA['artist_outperformance_multiplier']}x)")
print(f"   • Viral velocity ({VIRAL_CRITERIA['viral_engagement_threshold']:.1%} engagement ratio)")

print("=" * 70)
