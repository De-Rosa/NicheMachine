"""
predict_breakthrough_viral.py
-----------------------------------
Given a song name and artist, fetch data and predict BREAKTHROUGH viral potential.
Uses artist baselines and relative metrics (not absolute numbers).
"""

import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "./data"
LASTFM_API_KEY = "dc076f65a9c117cb52f6d8e88da750cc"
REQUEST_TIMEOUT = 10

# Load trained breakthrough model and features
MODEL_PATH = os.path.join(DATA_DIR, "lgb_breakthrough_detector.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "features_breakthrough.txt")

# Load artist baselines (needed for breakthrough detection!)
ARTIST_STATS_PATH = os.path.join(DATA_DIR, "merged_breakthrough_labeled.csv")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Breakthrough model not found at {MODEL_PATH}")
    print("   Run the pipeline first: python tiktok_pipeline_breakthrough_viral.py")
    exit(1)

print("📦 Loading breakthrough detection model...")
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, 'r') as f:
    required_features = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded with {len(required_features)} features")

# Load artist baselines
if os.path.exists(ARTIST_STATS_PATH):
    print("📊 Loading artist baselines...")
    full_data = pd.read_csv(ARTIST_STATS_PATH)
    artist_baselines = full_data.groupby("artist_name").agg({
        "lastfm_listeners": ["mean", "median"],
        "lastfm_playcount": ["mean", "median"]
    }).reset_index()
    artist_baselines.columns = ["artist_name", "artist_mean_listeners", "artist_median_listeners",
                                "artist_mean_playcount", "artist_median_playcount"]
    print(f"✅ Loaded baselines for {len(artist_baselines)} artists\n")
else:
    print("⚠️  No artist baselines found - will estimate from current track\n")
    artist_baselines = None


# -----------------------------
# FEATURE ENGINEERING FUNCTIONS
# -----------------------------

def engineer_acoustic_features(raw_tags_str):
    """Convert Last.fm tags to acoustic-like features"""
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


def fetch_lastfm_data(track_name, artist_name):
    """Fetch Last.fm data for a track including playcount, listeners, and tags"""
    print(f"🔍 Fetching Last.fm data for '{track_name}' by {artist_name}...")

    # Fetch track info
    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist_name)}&track={requests.utils.quote(track_name)}&format=json"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)

        if r.status_code != 200:
            print(f"❌ Last.fm API error: {r.status_code}")
            return None

        data = r.json()

        if "error" in data:
            print(f"❌ Track not found on Last.fm")
            return None

        track_info = data.get("track", {})

        lastfm_data = {
            "lastfm_listeners": int(track_info.get("listeners", 0))//2,
            "lastfm_playcount": int(track_info.get("playcount", 0))//2
        }

        print(f"   ✅ Found: {lastfm_data['lastfm_playcount']:,} plays, {lastfm_data['lastfm_listeners']:,} listeners")

    except Exception as e:
        print(f"❌ Error fetching track info: {str(e)}")
        return None

    # Fetch tags
    time.sleep(0.2)
    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=track.getTopTags&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist_name)}&track={requests.utils.quote(track_name)}&format=json"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)

        if r.status_code == 200:
            data = r.json()
            tags = data.get("toptags", {}).get("tag", [])

            if tags:
                tag_names = [t.get("name", "").lower() for t in tags[:30]]
                lastfm_data["raw_tags"] = ",".join(tag_names)
                lastfm_data["num_tags"] = len(tag_names)
                print(f"   ✅ Found {len(tag_names)} tags: {', '.join(tag_names[:5])}...")
            else:
                lastfm_data["raw_tags"] = ""
                lastfm_data["num_tags"] = 0
                print(f"   ⚠️  No tags found (using defaults)")
        else:
            lastfm_data["raw_tags"] = ""
            lastfm_data["num_tags"] = 0

    except Exception as e:
        print(f"   ⚠️  Could not fetch tags: {str(e)}")
        lastfm_data["raw_tags"] = ""
        lastfm_data["num_tags"] = 0

    return lastfm_data


def scrape_artist_top_tracks(artist_name):
    """Scrape artist's top tracks from Last.fm to compute baseline"""
    print(f"   🌐 Scraping artist's top tracks from Last.fm...")

    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getTopTracks&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist_name)}&limit=10&format=json"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)

        if r.status_code != 200:
            return None

        data = r.json()
        tracks = data.get("toptracks", {}).get("track", [])

        if not tracks or len(tracks) < 3:
            return None

        # Fetch detailed stats for top tracks
        track_stats = []
        for track in tracks[:10]:  # Top 10 tracks
            track_name = track.get("name")

            time.sleep(0.2)  # Rate limit
            try:
                track_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist_name)}&track={requests.utils.quote(track_name)}&format=json"
                track_r = requests.get(track_url, timeout=REQUEST_TIMEOUT)

                if track_r.status_code == 200:
                    track_data = track_r.json()
                    track_info = track_data.get("track", {})

                    listeners = int(track_info.get("listeners", 0))
                    playcount = int(track_info.get("playcount", 0))

                    if listeners > 0:
                        track_stats.append({
                            "listeners": listeners,
                            "playcount": playcount
                        })
            except:
                continue

        if len(track_stats) >= 3:
            # Calculate average from top tracks
            avg_listeners = np.mean([t["listeners"] for t in track_stats])
            avg_playcount = np.mean([t["playcount"] for t in track_stats])

            baseline = {
                "artist_mean_listeners": avg_listeners,
                "artist_mean_playcount": avg_playcount,
                "source": "scraped"
            }

            print(f"   ✅ Scraped {len(track_stats)} tracks → Baseline: {avg_listeners:,.0f} avg listeners")
            return baseline

        return None

    except Exception as e:
        print(f"   ❌ Scraping failed: {str(e)}")
        return None


def get_artist_baseline(artist_name):
    """Get artist's typical performance for comparison"""

    # Method 1: Check cached baselines from training data
    if artist_baselines is not None:
        artist_data = artist_baselines[artist_baselines["artist_name"].str.lower() == artist_name.lower()]

        if len(artist_data) > 0:
            baseline = {
                "artist_mean_listeners": artist_data.iloc[0]["artist_mean_listeners"],
                "artist_mean_playcount": artist_data.iloc[0]["artist_mean_playcount"],
                "source": "cached"
            }
            print(f"   📊 Artist baseline (cached): {baseline['artist_mean_listeners']:,.0f} avg listeners")
            return baseline

    # Method 2: Web scrape artist's top tracks
    print(f"   ℹ️  No cached baseline for '{artist_name}'")
    scraped_baseline = scrape_artist_top_tracks(artist_name)

    if scraped_baseline:
        return scraped_baseline

    # Method 3: No baseline available
    print(f"   ⚠️  Could not determine baseline (new/very obscure artist)")
    return None


def prepare_breakthrough_features(track_name, artist_name, lastfm_data, artist_baseline):
    """
    Prepare all features for breakthrough detection model.
    Key difference from old model: includes artist baseline comparisons!
    """
    print(f"\n🔧 Engineering breakthrough detection features...")

    # Extract base data
    listeners = lastfm_data.get("lastfm_listeners", 0)
    playcount = lastfm_data.get("lastfm_playcount", 0)
    raw_tags = lastfm_data.get("raw_tags", "")

    # Engineer acoustic features from tags
    acoustic_features = engineer_acoustic_features(raw_tags)

    # Calculate engagement metrics
    listener_play_ratio = listeners / (playcount + 1e-9) if playcount > 0 else 0.02

    # Simulate "early" signals (first week)
    early_listeners = listeners * 0.05
    early_playcount = playcount * 0.03
    early_velocity = early_playcount / (early_listeners + 1) if early_listeners > 0 else 0
    early_engagement_ratio = early_listeners / (early_playcount + 1e-9) if early_playcount > 0 else 0.02

    # Log transforms
    log_early_listeners = np.log1p(early_listeners)
    log_early_playcount = np.log1p(early_playcount)

    # BREAKTHROUGH-SPECIFIC: Artist baseline metrics
    if artist_baseline:
        artist_mean_listeners = artist_baseline["artist_mean_listeners"]
        artist_mean_playcount = artist_baseline["artist_mean_playcount"]
        baseline_source = artist_baseline.get("source", "unknown")

        # Key breakthrough metrics!
        listeners_vs_artist_mean = listeners / (artist_mean_listeners + 1)
        artist_mean_engagement = artist_mean_listeners / (artist_mean_playcount + 1e-9)
        early_engagement_vs_artist = early_engagement_ratio / (artist_mean_engagement + 1e-9)

        log_artist_baseline = np.log1p(artist_mean_listeners)

        print(f"   📈 Breakthrough Metrics ({baseline_source}):")
        print(f"      Outperformance: {listeners_vs_artist_mean:.2f}x artist average")
        print(f"      Early engagement: {early_engagement_vs_artist:.2f}x artist typical")
    else:
        # No baseline - estimate from current track (less accurate)
        print(f"   ⚠️  Using track as its own baseline (no artist history)")
        artist_mean_listeners = listeners * 0.5  # Assume this is above average
        listeners_vs_artist_mean = 2.0  # Assume moderate outperformance
        early_engagement_vs_artist = 1.0
        log_artist_baseline = np.log1p(artist_mean_listeners)

    # Feature interactions
    dance_x_energy = acoustic_features["fe_danceability"] * acoustic_features["fe_energy"]
    party_score = (acoustic_features["fe_energy"] + acoustic_features["fe_valence"]) / 2

    # Recency (assume new track)
    is_recent = 1

    # Combine all features
    features = {
        # Early signals
        "log_early_listeners": log_early_listeners,
        "log_early_playcount": log_early_playcount,
        "early_velocity": early_velocity,
        "early_engagement_ratio": early_engagement_ratio,
        "early_engagement_vs_artist": early_engagement_vs_artist,  # NEW!

        # Artist context
        "log_artist_baseline": log_artist_baseline,
        "listeners_vs_artist_mean": listeners_vs_artist_mean,  # NEW!

        # Engagement
        "listener_play_ratio": listener_play_ratio,

        # Acoustic features
        "fe_danceability": acoustic_features["fe_danceability"],
        "fe_energy": acoustic_features["fe_energy"],
        "fe_valence": acoustic_features["fe_valence"],
        "fe_tempo": acoustic_features["fe_tempo"],
        "fe_acousticness": acoustic_features["fe_acousticness"],

        # Interactions
        "dance_x_energy": dance_x_energy,
        "party_score": party_score,

        # Metadata
        "is_recent": is_recent,
        "fe_has_tags": acoustic_features["fe_has_tags"]
    }

    print(f"   ✅ Engineered {len(features)} features")

    return features


def predict_breakthrough_viral(track_name, artist_name):
    """
    Main function: Predict if track will be a BREAKTHROUGH viral hit.
    """
    print("=" * 70)
    print(f"🎵 BREAKTHROUGH VIRAL POTENTIAL PREDICTION")
    print("=" * 70)
    print(f"\nTrack:  {track_name}")
    print(f"Artist: {artist_name}\n")

    # Step 1: Fetch Last.fm data
    lastfm_data = fetch_lastfm_data(track_name, artist_name)

    if not lastfm_data:
        print("\n❌ Could not fetch data from Last.fm")
        return None

    # Step 2: Get artist baseline
    artist_baseline = get_artist_baseline(artist_name)

    # Step 3: Prepare features
    features = prepare_breakthrough_features(track_name, artist_name, lastfm_data, artist_baseline)

    # Step 4: Create DataFrame with all required features
    feature_df = pd.DataFrame([features])

    # Ensure all required features exist
    for feat in required_features:
        if feat not in feature_df.columns:
            feature_df[feat] = 0

    feature_df = feature_df[required_features]

    # Step 5: Make prediction
    print(f"\n🤖 Predicting breakthrough potential...")

    viral_probability = model.predict(feature_df, num_iteration=model.best_iteration)[0]

    # Interpret result
    print("\n" + "=" * 70)
    print("📊 BREAKTHROUGH PREDICTION RESULTS")
    print("=" * 70)

    print(f"\n🎯 Breakthrough Probability: {viral_probability:.1%}")

    if viral_probability >= 0.7:
        verdict = "🔥 HIGH BREAKTHROUGH POTENTIAL"
        confidence = "Strong"
        recommendation = "This track significantly outperforms artist expectations! Strong viral indicators."
    elif viral_probability >= 0.5:
        verdict = "⚡ MODERATE BREAKTHROUGH POTENTIAL"
        confidence = "Good"
        recommendation = "Track shows promise of exceeding typical performance for this artist."
    elif viral_probability >= 0.3:
        verdict = "💫 MILD BREAKTHROUGH POTENTIAL"
        confidence = "Uncertain"
        recommendation = "Some positive signals, but not a clear breakthrough."
    else:
        verdict = "❄️  LOW BREAKTHROUGH POTENTIAL"
        confidence = "Low"
        recommendation = "Track likely performs at or below artist's typical level."

    print(f"\n{verdict}")
    print(f"Confidence: {confidence}")
    print(f"\n💡 Recommendation:")
    print(f"   {recommendation}")

    # Show contributing factors
    print(f"\n📈 Breakthrough Indicators:")

    if artist_baseline:
        baseline_source = artist_baseline.get("source", "unknown")
        print(f"   ℹ️  Artist baseline: {baseline_source}")

        if features['listeners_vs_artist_mean'] >= 2.0:
            print(f"   ✅ Strong outperformance: {features['listeners_vs_artist_mean']:.1f}x artist average")
        elif features['listeners_vs_artist_mean'] >= 1.5:
            print(f"   ⚡ Moderate outperformance: {features['listeners_vs_artist_mean']:.1f}x artist average")
        else:
            print(f"   ❄️  Below 2x threshold: {features['listeners_vs_artist_mean']:.1f}x artist average")

        if features['early_engagement_vs_artist'] >= 1.5:
            print(f"   ✅ Early catch-on: {features['early_engagement_vs_artist']:.1f}x artist typical engagement")
    else:
        print(f"   ⚠️  Unknown artist - baseline unavailable (prediction less reliable)")

    if features['listener_play_ratio'] >= 0.03:
        print(f"   ✅ High viral velocity: {features['listener_play_ratio']:.3f} engagement ratio")

    if features['fe_danceability'] > 0.7:
        print(f"   ✅ Highly danceable ({features['fe_danceability']:.2f}) - good for TikTok")
    if features['party_score'] > 0.7:
        print(f"   ✅ Party vibes ({features['party_score']:.2f}) - social sharing potential")

    print("\n" + "=" * 70)

    return {
        "viral_probability": viral_probability,
        "verdict": verdict,
        "confidence": confidence,
        "features": features,
        "lastfm_data": lastfm_data,
        "artist_baseline": artist_baseline
    }


# -----------------------------
# INTERACTIVE MODE
# -----------------------------

if __name__ == "__main__":
    print("🎵 Breakthrough Viral Predictor")
    print("=" * 70)
    print("\nPredicts if a track will BREAKTHROUGH beyond artist's typical performance\n")

    track_name = "Formidable"
    artist_name = "Stromae"

    if not track_name or not artist_name:
        print("\n❌ Both track name and artist name are required")
        exit(1)

    print()

    result = predict_breakthrough_viral(track_name, artist_name)

    if result:
        save = input("\n💾 Save prediction? (y/n): ").strip().lower()
        if save == 'y':
            results_file = os.path.join(DATA_DIR, "breakthrough_predictions_log.csv")

            prediction_record = {
                "timestamp": datetime.now().isoformat(),
                "track_name": track_name,
                "artist_name": artist_name,
                "breakthrough_probability": result["viral_probability"],
                "verdict": result["verdict"],
                "listeners": result["lastfm_data"]["lastfm_listeners"],
                "vs_artist_mean": result["features"].get("listeners_vs_artist_mean", None)
            }

            if os.path.exists(results_file):
                log_df = pd.read_csv(results_file)
                log_df = pd.concat([log_df, pd.DataFrame([prediction_record])], ignore_index=True)
            else:
                log_df = pd.DataFrame([prediction_record])

            log_df.to_csv(results_file, index=False)
            print(f"✅ Saved to {results_file}")