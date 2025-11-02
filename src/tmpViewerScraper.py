import requests

API_KEY = "API_KEY"  
BASE = "https://ws.audioscrobbler.com/2.0/"

def get_lastfm_listeners(artist_name: str):
    params = {
        "method": "artist.getinfo",
        "artist": artist_name,
        "api_key": API_KEY,
        "format": "json"
    }
    r = requests.get(BASE, params=params, timeout=5)
    data = r.json()
    if "artist" in data and "stats" in data["artist"]:
        listeners = int(data["artist"]["stats"]["listeners"])
        plays = int(data["artist"]["stats"]["playcount"])
        return {"listeners": listeners, "plays": plays}
    return None

def is_niche_artist(artist_name: str, listener_threshold=1_000_000):
    info = get_lastfm_listeners(artist_name)
    if not info:
        return None
    return info["listeners"] < listener_threshold, info

# Example
result, stats = is_niche_artist("Arca")
print(stats)  # {'listeners': 420000, 'plays': 12500000}
print("Niche?", result)
