import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

DATA_DIR = "./data"
RANDOM_STATE = 42

MIN_TOTAL_PLAYCOUNT = 1_000_000
MIN_LISTENERS = 50_000
LISTENER_TO_PLAY_RATIO_MIN = 0.02

required_files = {
    "lastfm_metadata.csv": "Last.fm metadata",
    "mbids.csv": "MusicBrainz IDs",
    "acousticbrainz_features.csv": "AcousticBrainz features"
}

missing_files = []
for filename, description in required_files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        pass
    else:
        missing_files.append(filename)

if missing_files:
    exit(1)

try:
    kag = pd.read_csv("tiktok.csv")
    kag.columns = [c.lower() for c in kag.columns]

    track_col = "track_name" if "track_name" in kag.columns else "track"
    artist_col = "artist_name" if "artist_name" in kag.columns else "artist"
    date_col = "release_date"

    kag = kag.rename(columns={track_col: "track_name", artist_col: "artist_name"})
    kag[date_col] = pd.to_datetime(kag[date_col], errors="coerce")

    kag["combo"] = kag["track_name"].str.lower().fillna('') + " - " + kag["artist_name"].str.lower().fillna('')
    unique_tracks = kag.drop_duplicates(subset=["combo"])[["track_name", "artist_name", date_col]].dropna()

except FileNotFoundError:
    exit(1)

lastfm_meta = pd.read_csv(os.path.join(DATA_DIR, "lastfm_metadata.csv"))

mbid_meta = pd.read_csv(os.path.join(DATA_DIR, "mbids.csv"))

acoustic_meta = pd.read_csv(os.path.join(DATA_DIR, "acousticbrainz_features.csv"))


merged = unique_tracks.copy()

if len(lastfm_meta) > 0:
    merged = merged.merge(lastfm_meta, on=["track_name", "artist_name"], how="left")
if len(mbid_meta) > 0:
    merged = merged.merge(mbid_meta, on=["track_name", "artist_name"], how="left")
if len(acoustic_meta) > 0:
    merged = merged.merge(acoustic_meta, on=["track_name", "artist_name"], how="left")

merged.to_csv(os.path.join(DATA_DIR, "merged_full.csv"), index=False)


def calculate_viral_label(row):
    playcount = row.get("lastfm_playcount", 0)
    listeners = row.get("lastfm_listeners", 0)
    release_date = pd.to_datetime(row.get("release_date"))

    try:
        playcount = float(playcount) if pd.notna(playcount) else 0
        listeners = float(listeners) if pd.notna(listeners) else 0
    except:
        playcount = 0
        listeners = 0

    if playcount < MIN_TOTAL_PLAYCOUNT or listeners < MIN_LISTENERS:
        return 0

    engagement_ratio = listeners / (playcount + 1e-9)
    if engagement_ratio < LISTENER_TO_PLAY_RATIO_MIN:
        return 0

    if pd.notna(release_date):
        days_since_release = (pd.Timestamp.now() - release_date).days
        is_recent = days_since_release < 730

        if is_recent:
            viral_threshold_playcount = MIN_TOTAL_PLAYCOUNT * 0.7
            viral_threshold_listeners = MIN_LISTENERS * 0.7
        else:
            viral_threshold_playcount = MIN_TOTAL_PLAYCOUNT * 1.5
            viral_threshold_listeners = MIN_LISTENERS * 1.5

        if playcount >= viral_threshold_playcount and listeners >= viral_threshold_listeners:
            return 1
    else:
        if playcount >= MIN_TOTAL_PLAYCOUNT and listeners >= MIN_LISTENERS:
            return 1

    return 0

merged["viral_label"] = merged.apply(calculate_viral_label, axis=1)

merged["listener_play_ratio"] = merged["lastfm_listeners"] / (merged["lastfm_playcount"] + 1e-9)
merged["days_since_release"] = (pd.Timestamp.now() - pd.to_datetime(merged["release_date"])).dt.days

merged.to_csv(os.path.join(DATA_DIR, "merged_labeled.csv"), index=False)


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
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "is_unbalance": True,
    "seed": RANDOM_STATE,
    "device": "cpu"
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_test],
    valid_names=["train", "test"],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)


y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:,.0f}")

model_path = os.path.join(DATA_DIR, "lgb_viral_predictor.pkl")
joblib.dump(model, model_path)

feature_list_path = os.path.join(DATA_DIR, "feature_list.txt")
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_cols))

print("\n🏁 Pipeline complete! Model ready for predictions.")
