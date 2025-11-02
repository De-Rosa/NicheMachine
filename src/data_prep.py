import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = "./data"
LASTFM_API_KEY = "API_KEY"
REQUEST_TIMEOUT = 10

MODEL_PATH = os.path.join(DATA_DIR, "lgb_breakthrough_detector.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "features_breakthrough.txt")

ARTIST_STATS_PATH = os.path.join(DATA_DIR, "merged_breakthrough_labeled.csv")

if not os.path.exists(MODEL_PATH):
    print("No model exists")
    exit(1)

print("📦 Loading breakthrough detection model...")
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, 'r') as f:
    required_features = [line.strip() for line in f.readlines()]

if os.path.exists(ARTIST_STATS_PATH):
    full_data = pd.read_csv(ARTIST_STATS_PATH)
    artist_baselines = full_data.groupby("artist_name").agg({
        "lastfm_listeners": ["mean", "median"],
        "lastfm_playcount": ["mean", "median"]
    }).reset_index()
    artist_baselines.columns = ["artist_name", "artist_mean_listeners", "artist_median_listeners",
                                "artist_mean_playcount", "artist_median_playcount"]
else:
    artist_baselines = None


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
    
    dance_indicators = {
        "dance", "electronic", "edm", "house", "techno", "disco", "funk",
        "dancehall", "electro", "trance", "dubstep", "drum and bass",
        "dance-pop", "electropop", "synthpop", "dance-punk"
    }
    danceability_score = len(tags & dance_indicators) / 5.0
    danceability = min(0.95, max(0.2, 0.4 + danceability_score * 0.5))

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


    # Fetch track info
    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist_name)}&track={requests.utils.quote(track_name)}&format=json"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)

        if r.status_code != 200:
            print(f"API error: {r.status_code}")
            return None

        data = r.json()

        if "error" in data:
            print("Track not found on Last.fm")
            return None

        track_info = data.get("track", {})

        lastfm_data = {
            "lastfm_listeners": int(track_info.get("listeners", 0))//2,
            "lastfm_playcount": int(track_info.get("playcount", 0))//2
        }


    except Exception as e:
        print(f"Error fetching track info: {str(e)}")
        return None

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
            else:
                lastfm_data["raw_tags"] = ""
                lastfm_data["num_tags"] = 0
        else:
            lastfm_data["raw_tags"] = ""
            lastfm_data["num_tags"] = 0

    except Exception as e:
        lastfm_data["raw_tags"] = ""
        lastfm_data["num_tags"] = 0

    return lastfm_data


def scrape_artist_top_tracks(artist_name):

    try:
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getTopTracks&api_key={LASTFM_API_KEY}&artist={requests.utils.quote(artist_name)}&limit=10&format=json"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)

        if r.status_code != 200:
            return None

        data = r.json()
        tracks = data.get("toptracks", {}).get("track", [])

        if not tracks or len(tracks) < 3:
            return None

        track_stats = []
        for track in tracks[:10]: 
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
            avg_listeners = np.mean([t["listeners"] for t in track_stats])
            avg_playcount = np.mean([t["playcount"] for t in track_stats])

            baseline = {
                "artist_mean_listeners": avg_listeners,
                "artist_mean_playcount": avg_playcount,
                "source": "scraped"
            }

            return baseline

        return None

    except Exception as e:
        print(f"Scraping failed: {str(e)}")
        return None


def get_artist_baseline(artist_name):

    if artist_baselines is not None:
        artist_data = artist_baselines[artist_baselines["artist_name"].str.lower() == artist_name.lower()]

        if len(artist_data) > 0:
            baseline = {
                "artist_mean_listeners": artist_data.iloc[0]["artist_mean_listeners"],
                "artist_mean_playcount": artist_data.iloc[0]["artist_mean_playcount"],
                "source": "cached"
            }
            return baseline
    scraped_baseline = scrape_artist_top_tracks(artist_name)

    if scraped_baseline:
        return scraped_baseline

    #No baseline available
    return None


def prepare_breakthrough_features(track_name, artist_name, lastfm_data, artist_baseline):


    listeners = lastfm_data.get("lastfm_listeners", 0)
    playcount = lastfm_data.get("lastfm_playcount", 0)
    raw_tags = lastfm_data.get("raw_tags", "")

    acoustic_features = engineer_acoustic_features(raw_tags)

    listener_play_ratio = listeners / (playcount + 1e-9) if playcount > 0 else 0.02

    early_listeners = listeners * 0.05
    early_playcount = playcount * 0.03
    early_velocity = early_playcount / (early_listeners + 1) if early_listeners > 0 else 0
    early_engagement_ratio = early_listeners / (early_playcount + 1e-9) if early_playcount > 0 else 0.02

    log_early_listeners = np.log1p(early_listeners)
    log_early_playcount = np.log1p(early_playcount)

    if artist_baseline:
        artist_mean_listeners = artist_baseline["artist_mean_listeners"]
        artist_mean_playcount = artist_baseline["artist_mean_playcount"]
        baseline_source = artist_baseline.get("source", "unknown")

        listeners_vs_artist_mean = listeners / (artist_mean_listeners + 1)
        artist_mean_engagement = artist_mean_listeners / (artist_mean_playcount + 1e-9)
        early_engagement_vs_artist = early_engagement_ratio / (artist_mean_engagement + 1e-9)

        log_artist_baseline = np.log1p(artist_mean_listeners)

    else:
        artist_mean_listeners = listeners * 0.5  
        listeners_vs_artist_mean = 2.0  
        early_engagement_vs_artist = 1.0
        log_artist_baseline = np.log1p(artist_mean_listeners)

    dance_x_energy = acoustic_features["fe_danceability"] * acoustic_features["fe_energy"]
    party_score = (acoustic_features["fe_energy"] + acoustic_features["fe_valence"]) / 2

    is_recent = 1

    features = {
        "log_early_listeners": log_early_listeners,
        "log_early_playcount": log_early_playcount,
        "early_velocity": early_velocity,
        "early_engagement_ratio": early_engagement_ratio,
        "early_engagement_vs_artist": early_engagement_vs_artist,  # NEW!

        "log_artist_baseline": log_artist_baseline,
        "listeners_vs_artist_mean": listeners_vs_artist_mean,  # NEW!

        "listener_play_ratio": listener_play_ratio,

        "fe_danceability": acoustic_features["fe_danceability"],
        "fe_energy": acoustic_features["fe_energy"],
        "fe_valence": acoustic_features["fe_valence"],
        "fe_tempo": acoustic_features["fe_tempo"],
        "fe_acousticness": acoustic_features["fe_acousticness"],

        "dance_x_energy": dance_x_energy,
        "party_score": party_score,

        "is_recent": is_recent,
        "fe_has_tags": acoustic_features["fe_has_tags"]
    }


    return features


def predict_breakthrough_viral(track_name, artist_name):

    lastfm_data = fetch_lastfm_data(track_name, artist_name)

    if not lastfm_data:
        print("Could not fetch data from Last.fm")
        return None

    artist_baseline = get_artist_baseline(artist_name)

    features = prepare_breakthrough_features(track_name, artist_name, lastfm_data, artist_baseline)

    feature_df = pd.DataFrame([features])

    for feat in required_features:
        if feat not in feature_df.columns:
            feature_df[feat] = 0

    feature_df = feature_df[required_features]

    viral_probability = model.predict(feature_df, num_iteration=model.best_iteration)[0]


    if viral_probability >= 0.7:
        verdict = "HIGH"
        confidence = "Strong"
        recommendation = "This track significantly outperforms artist expectations! Strong viral indicators."
    elif viral_probability >= 0.5:
        verdict = "MODERATE"
        confidence = "Good"
        recommendation = "Track shows promise of exceeding typical performance for this artist."
    elif viral_probability >= 0.3:
        verdict = "MILD"
        confidence = "Uncertain"
        recommendation = "Some positive signals, but not a clear breakthrough."
    else:
        verdict =  "LOW"
        confidence = "Low"
        recommendation = "Track likely performs at or below artist's typical level."



    if artist_baseline:
        baseline_source = artist_baseline.get("source", "unknown")


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
    track_name = "TITLE"
    artist_name = "ARTIST"

    if not track_name or not artist_name:
        print("Both track name and artist name are required")
        exit(1)

    print()

    result = predict_breakthrough_viral(track_name, artist_name)

    if result:
        save = input("Save prediction? (y/n): ").strip().lower()
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

            print(f"Saved to {results_file}")
