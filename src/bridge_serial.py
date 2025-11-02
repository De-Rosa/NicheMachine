import serial
import pandas as pd
from flask import Flask
from flask_socketio import SocketIO

# ---------- Configuration ----------
SERIAL_PORT = "/dev/tty.usbmodemF412FA9FE0602"  # Change to your Arduino port
BAUD_RATE = 9600
FLASK_PORT = 5000

# ---------- Flask + SocketIO ----------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return "Arduino USB Serial Bridge Running"

# ---------- Load ranked.csv ----------
ranked_csv_path = "ranked.csv"
try:
    df_ranked = pd.read_csv(ranked_csv_path)
    df_ranked = df_ranked.drop_duplicates(subset=["song_title", "artist_name"])
    df_ranked = df_ranked.sort_values(by="viral_probability", ascending=False).reset_index(drop=True)
    print(f"✅ Loaded {len(df_ranked)} unique tracks from {ranked_csv_path}")
    print(df_ranked)
except Exception as e:
    print(f"❌ Failed to load {ranked_csv_path}: {e}")
    df_ranked = pd.DataFrame(columns=["song_title", "artist_name", "viral_probability", "verdict", "confidence"])

# ---------- Load preload.csv for playcount & genre ----------
preload_csv_path = "preload.csv"
try:
    df_preload = pd.read_csv(preload_csv_path)
    # Normalize columns for safer joining
    df_preload = df_preload.rename(columns={"tags": "genre"})
    print(f"✅ Loaded {len(df_preload)} entries from {preload_csv_path}")
except Exception as e:
    print(f"❌ Failed to load {preload_csv_path}: {e}")
    df_preload = pd.DataFrame(columns=["song_title", "artist_name", "playcount", "genre"])

# ---------- Merge the two datasets ----------
if not df_ranked.empty and not df_preload.empty:
    df = pd.merge(
        df_ranked,
        df_preload[["song_title", "artist_name", "playcount", "genre"]],
        on=["song_title", "artist_name"],
        how="left"
    )
else:
    df = df_ranked.copy()

# Fill missing playcount or genre values gracefully
df["playcount"] = df.get("playcount", pd.Series()).fillna(0).astype(int)
df["genre"] = df.get("genre", pd.Series()).fillna("Unknown")

print(f"🎵 Final merged dataset: {len(df)} tracks with playcount and genre info.")

# ---------- State ----------
current_index = 0
web_is_idle = True  # prevents overlapping triggers

# ---------- SocketIO Event ----------
@socketio.on("client_idle")
def on_client_idle():
    """Client notifies that it is ready for the next song."""
    global web_is_idle
    web_is_idle = True
    print("✅ Client is now idle and ready for next trigger.")

# ---------- Serial Reading Task ----------
def read_from_arduino():
    global current_index, web_is_idle

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"🔌 Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        print(f"❌ Failed to connect to Arduino: {e}")
        return

    while True:
        try:
            line = ser.readline().decode().strip()
            if not line:
                continue

            print(f"📩 Received from Arduino: {line}")

            if line == "TRIGGER":
                if not web_is_idle:
                    print("⚠️ Webpage busy, ignoring trigger.")
                    continue

                if df.empty:
                    print("⚠️ No ranked tracks available.")
                    continue

                # Get next track
                track = df.iloc[current_index]
                current_index = (current_index + 1) % len(df)
                web_is_idle = False  # mark as busy

                # Safely extract data
                song = track.get("song_title", "Unknown Song")
                artist = track.get("artist_name", "Unknown Artist")
                confidence = track.get("confidence", "N/A")
                playcount = track.get("playcount", 0)
                genre = track.get("genre", "Unknown")

                subtitle_parts = [
                    f"Confidence: {confidence}",
                    f"Playcount: {playcount:,}"
                ]
                if genre and genre != "Unknown" and str(genre).strip() != "[]":
                    subtitle_parts.append(f"Genre: {genre}")

                subtitle = " | ".join(subtitle_parts)

                print(f"🎶 Sending {song} by {artist}")
                socketio.emit("trigger", {
                    "song": song,
                    "artist": artist,
                    "subtitle": subtitle
                })

        except Exception as e:
            print(f"⚠️ Error reading serial: {e}")

# ---------- Start Serial Reading ----------
socketio.start_background_task(read_from_arduino)

# ---------- Main ----------
if __name__ == "__main__":
    print(f"🚀 Flask-SocketIO server running on http://0.0.0.0:{FLASK_PORT}")
    socketio.run(app, host="0.0.0.0", port=FLASK_PORT)

