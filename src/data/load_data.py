__all__ = ["SpotipyAgent"]

import spotipy
import os
import asyncio
import unicodedata
import re
import requests
from spotipy.oauth2 import SpotifyClientCredentials
from TikTokApi import TikTokApi
from dotenv import load_dotenv
import logging
from utils.data import (
    has_non_english_letters,
    normalize_song_info,
    print_debug,
    fetch_music_details,
    print_videos_with_music_features,
    clean_name
)

logging.disable(logging.CRITICAL)

class LastFmAcousticBrainzAgent:
    def __init__(self):
        load_dotenv()
        self.lastfm_key = os.getenv('LASTFM_API_KEY')
        self.lastfm_url = "http://ws.audioscrobbler.com/2.0/"
        self.acousticbrainz_url = "https://acousticbrainz.org/api/v1/"

    def _lastfm_request(self, params: dict) -> dict:
        params['api_key'] = self.lastfm_key
        params['format'] = 'json'
        res = requests.get(self.lastfm_url, params=params)
        res.raise_for_status()
        return res.json()

    def get_track_info(self, track_name: str, artist_name: str) -> dict | None:
        """Fetch Last.fm metadata and AcousticBrainz features (if MBID available)."""
        params = {
            'method': 'track.getInfo',
            'track': track_name,
            'artist': artist_name
        }
        data = self._lastfm_request(params)
        track = data.get('track')
        if not track:
            return None

        result = {
            "track_name": track.get('name'),
            "artist_name": track.get('artist', {}).get('name'),
            "album": track.get('album', {}).get('title', 'Unknown'),
            "url": track.get('url'),
            "listeners": track.get('listeners'),
            "playcount": track.get('playcount'),
            "tags": [t['name'] for t in track.get('toptags', {}).get('tag', [])],
            "audio_features": None
        }

        mbid = track.get('mbid')
        if mbid:
            features = self.get_acousticbrainz_features(mbid)
            if features:
                result['audio_features'] = features

        return result

    def get_acousticbrainz_features(self, mbid: str) -> dict | None:
        """Retrieve audio analysis features from AcousticBrainz using MusicBrainz ID."""
        try:
            url = f"{self.acousticbrainz_url}{mbid}/low-level"
            res = requests.get(url)
            if res.status_code != 200:
                return None
            data = res.json()

            high_level_url = f"{self.acousticbrainz_url}{mbid}/high-level"
            res_high = requests.get(high_level_url)
            if res_high.status_code == 200:
                data.update(res_high.json())

            # Extract selected key features
            return {
                "bpm": data.get("rhythm", {}).get("bpm"),
                "danceability": data.get("highlevel", {}).get("danceability", {}).get("value"),
                "mood_happy": data.get("highlevel", {}).get("mood_happy", {}).get("value"),
                "mood_sad": data.get("highlevel", {}).get("mood_sad", {}).get("value"),
                "acousticness": data.get("highlevel", {}).get("mood_acoustic", {}).get("value"),
                "instrumental": data.get("highlevel", {}).get("mood_instrumental", {}).get("value"),
                "genre": data.get("highlevel", {}).get("genre_dortmund", {}).get("value")
            }

        except Exception as e:
            print(f"AcousticBrainz error for MBID {mbid}: {e}")
            return None
class TiktokAgent:
    def __init__(self):
        pass

    def get_videos(self, limit: int = 50):
        return asyncio.run(self.fetch_videos(max_videos=limit))

    async def fetch_videos(self, max_videos=50):
        ms_token = os.getenv("MS_TOKEN")
        async with TikTokApi() as api:
            print_debug("Creating playwright session...")
            await api.create_sessions(
                ms_tokens=[ms_token],
                num_sessions=1,
                sleep_after=3,
                browser=os.getenv("TIKTOK_BROWSER", "chromium")
            )
            print_debug("Created playwright session!")

            momentum_videos = []
            fetched = 0
            batch_size = 30
            seen_ids = set()

            while fetched < max_videos:
                remaining = max_videos - fetched
                count = min(batch_size, remaining)
                print_debug(f"Fetching next {count} videos...")

                new_videos_in_batch = 0

                async for video in api.trending.videos(count=count):
                    try:
                        video_dict = video.as_dict
                        video_id = video_dict.get('id')

                        if not video_id or video_id in seen_ids:
                            continue
                        seen_ids.add(video_id)

                        if video_dict.get('liveRoomInfo'):
                            continue

                        stats = video_dict.get('stats', {})
                        if not stats or stats.get('playCount', 0) == 0:
                            continue

                        likes = stats.get('diggCount', 0)
                        views = stats.get('playCount', 0)
                        comments = stats.get('commentCount', 0)
                        shares = stats.get('shareCount', 0)
                        engagement_rate = (likes + comments + shares) / views

                        music_info = video_dict.get('music', {})
                        author_info = video_dict.get('author', {})
                        song_title = music_info.get('title', 'Unknown')
                        artist_name = music_info.get('authorName', 'Unknown')
                        music_id = music_info.get('id', 'N/A')


                        if 'original' in song_title.lower() or 'original' in artist_name.lower():
                            continue

                        song_title, artist_name = normalize_song_info(song_title, artist_name)
                        if song_title == 'Unknown' or artist_name == 'Unknown':
                            continue

                        if 'Original' in song_title or 'Original' in artist_name:
                            continue
                        
                        print(song_title)

                        video_data = {
                            'video_id': video_id,
                            'description': video_dict.get('desc', '')[:100],
                            'likes': likes,
                            'views': views,
                            'comments': comments,
                            'shares': shares,
                            'engagement_rate': round(engagement_rate, 4),
                            'song_title': song_title,
                            'artist_name': artist_name,
                            'music_id': music_id,
                        }

                        momentum_videos.append(video_data)
                        fetched += 1
                        new_videos_in_batch += 1

                        if fetched >= max_videos:
                            break

                    except Exception as e:
                        print(f"⊗ Error: {e}")
                        continue

                # If this batch added no new videos, stop to avoid endless loop
                if new_videos_in_batch == 0:
                    print_debug("No new valid videos found in this batch. Stopping fetch.")
                    break

                if fetched < max_videos:
                    await asyncio.sleep(2)  # polite delay between batches

            print_debug(f"Fetched total {len(momentum_videos)} unique videos.")
            return momentum_videos


if __name__ == "__main__":
    tiktok_agent = TiktokAgent()
    music_agent = LastFmAcousticBrainzAgent()

    momentum_videos = tiktok_agent.get_videos(limit=200)

    results = fetch_music_details(momentum_videos, music_agent)

    print_videos_with_music_features(results)
