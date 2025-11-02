from data.load_data import get_videos
import pandas as pd

def preload():
    videos = get_videos()
    df = pd.DataFrame(videos)

    df.to_csv("videos.csv", index=False)
    print(f"Saved {len(videos)} records to enriched_videos.csv")

if __name__ == "__main__":
    preload()
