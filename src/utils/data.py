import unicodedata
import re

DEBUG = True

def print_debug(message: str) -> None:
    if DEBUG:
        print(f"DEBUG: {message}")


def has_non_english_letters(text: str) -> bool:
    for char in text:
        if not char.isalpha():
            continue
        base_char = unicodedata.normalize('NFKD', char)[0]
        if not re.match(r'[A-Za-z]', base_char):
            return True
    return False


def normalize_song_info(title: str, artist: str) -> tuple[str, str]:
    title = str(title).strip()
    artist = str(artist).strip()

    original_sound_patterns = [
        r'^original\s+sound\s*-?\s*',
        r'^son\s+original\s*-?\s*',
        r'^suara\s+asli\s*-?\s*',
        r'^оригинальный\s+звук\s*-?\s*',
        r'^som\s+original\s*-?\s*',
        r'^\[original\s+sound\]',
    ]

    title_lower = title.lower()
    for pattern in original_sound_patterns:
        if re.search(pattern, title_lower):
            return 'Original Sound', 'Original Sound'

    title = re.sub(r'^original\s+sound\s*-\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^son\s+original\s*-\s*', '', title, flags=re.IGNORECASE)

    title = title.replace('  ', ' ').strip()
    artist = artist.replace('  ', ' ').strip()
    artist = re.sub(r'^@+', '', artist)

    if title and not any(c.isupper() for c in title):
        title = title.title()
    if artist and not any(c.isupper() for c in artist):
        artist = artist.title()

    if not title or title.lower() in ['unknown', 'n/a', 'none']:
        title = 'Unknown'
    if not artist or artist.lower() in ['unknown', 'n/a', 'none']:
        artist = 'Unknown'
    if has_non_english_letters(artist.lower()) or has_non_english_letters(title.lower()):
        artist = 'Unknown'
        title = 'Unknown'

    if "original" in title.lower():
        title = 'Unknown'

    return title, artist


def clean_name(s):
    """Normalize and clean track/artist names for search."""
    s = unicodedata.normalize('NFKD', s)
    s = re.sub(r'[\W_]+', ' ', s).strip().lower()
    return s

def fetch_music_details(momentum_videos: list[dict], agent) -> list[dict]:
    """Fetch both Last.fm metadata and AcousticBrainz features."""
    enriched = []

    for video in momentum_videos:
        track_name = video.get('song_title')
        artist_name = video.get('artist_name')

        if not track_name or not artist_name:
            continue

        track_data = agent.get_track_info(track_name, artist_name)
        if not track_data:
            print(f"No match for {track_name} - {artist_name}")
            continue

        video_copy = video.copy()
        video_copy.update({
            "lastfm_link": track_data.get("url"),
            "lastfm_album": track_data.get("album"),
            "listeners": track_data.get("listeners"),
            "playcount": track_data.get("playcount"),
            "tags": track_data.get("tags"),
            "audio_features": track_data.get("audio_features")
        })
        enriched.append(video_copy)

    return enriched


def print_videos_with_music_features(videos: list[dict]):
    """Pretty-print enriched music data."""
    for idx, v in enumerate(videos, 1):
        print(f"#{idx}: {v['likes']:,} likes | {v['views']:,} views | {v['engagement_rate']:.2%}")
        print(f"  🎵 {v['song_title']} - {v['artist_name']}")
        print(f"  💿 Album: {v.get('lastfm_album', 'Unknown')}")
        print(f"  🔗 Last.fm: {v.get('lastfm_link', 'N/A')}")
        print(f"  👂 Listeners: {v.get('listeners', '0')}")
        print(f"  ▶️ Playcount: {v.get('playcount', '0')}")
        print(f"  🏷️ Tags: {', '.join(v.get('tags', []))}")
        af = v.get("audio_features")
        if af:
            print("  🎚️ Audio Features:")
            for k, val in af.items():
                print(f"    - {k}: {val}")
        else:
            print("  🎚️ Audio Features: unavailable")
        print("\n")

def print_videos_with_lastfm_features(videos: list[dict]):
    """
    Print enriched video info including Last.fm details.
    """
    for idx, v in enumerate(videos, 1):
        print(f"#{idx}: {v['likes']:,} likes | {v['views']:,} views | {v['engagement_rate']:.2%}")
        print(f"  🎵 {v['song_title']} - {v['artist_name']}")
        print(f"  🆔 {v['video_id']}")
        print(f"  ✍️ {v['description']}")
        print(f"  🔗 Last.fm: {v.get('lastfm_link', 'N/A')}")
        print(f"  💿 Album: {v.get('lastfm_album', 'Unknown')}")
        print(f"  👂 Listeners: {v.get('lastfm_listeners', '0')}")
        print(f"  ▶️ Playcount: {v.get('lastfm_playcount', '0')}")
        print("\n")
