"""
tiktok_import_and_train.py
-----------------------------------
Import pre-downloaded data and train viral prediction model.
Skips all API calls - uses cached data only.
"""

import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "./data"
RANDOM_STATE = 42

# Last.fm-based thresholds
MIN_TOTAL_PLAYCOUNT = 1_000_000  # 1M total plays
MIN_LISTENERS = 50_000  # 50k unique listeners
LISTENER_TO_PLAY_RATIO_MIN = 0.02  # At least 2% listener/play ratio

# -----------------------------
# 1. Check for cached data files
# -----------------------------
print("📂 Checking for cached data files...")

required_files = {
    "lastfm_metadata.csv": "Last.fm metadata",
    "mbids.csv": "MusicBrainz IDs",
    "acousticbrainz_features.csv": "AcousticBrainz features"
}

missing_files = []
for filename, description in required_files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"✅ Found {description}: {filename}")
    else:
        print(f"❌ Missing {description}: {filename}")
        missing_files.append(filename)

if missing_files:
    print(f"\n⚠️  Missing {len(missing_files)} required file(s).")
    print("   Please run the full pipeline first to download data.")
    exit(1)

print("\n✅ All required data files found!")

# -----------------------------
# 2. Load TikTok dataset
# -----------------------------
print("\n📥 Loading TikTok Trending Tracks dataset...")
try:
    kag = pd.read_csv("tiktok.csv")
    kag.columns = [c.lower() for c in kag.columns]

    # Standardize column names
    track_col = "track_name" if "track_name" in kag.columns else "track"
    artist_col = "artist_name" if "artist_name" in kag.columns else "artist"
    date_col = "release_date"

    kag = kag.rename(columns={track_col: "track_name", artist_col: "artist_name"})
    kag[date_col] = pd.to_datetime(kag[date_col], errors="coerce")

    kag["combo"] = kag["track_name"].str.lower().fillna('') + " - " + kag["artist_name"].str.lower().fillna('')
    unique_tracks = kag.drop_duplicates(subset=["combo"])[["track_name", "artist_name", date_col]].dropna()

    print(f"✅ Loaded {len(unique_tracks)} unique tracks from TikTok dataset.")
except FileNotFoundError:
    print("❌ tiktok.csv not found!")
    exit(1)

# -----------------------------
# 3. Load cached metadata
# -----------------------------
print("\n📖 Loading cached metadata...")

lastfm_meta = pd.read_csv(os.path.join(DATA_DIR, "lastfm_metadata.csv"))
print(f"✅ Loaded {len(lastfm_meta)} Last.fm tracks")

mbid_meta = pd.read_csv(os.path.join(DATA_DIR, "mbids.csv"))
print(f"✅ Loaded {len(mbid_meta)} MBIDs ({mbid_meta['mbid'].notna().sum()} valid)")

acoustic_meta = pd.read_csv(os.path.join(DATA_DIR, "acousticbrainz_features.csv"))
print(f"✅ Loaded {len(acoustic_meta)} AcousticBrainz features")

# -----------------------------
# 4. Merge all datasets
# -----------------------------
print("\n🔄 Merging datasets...")

merged = unique_tracks.copy()

if len(lastfm_meta) > 0:
    merged = merged.merge(lastfm_meta, on=["track_name", "artist_name"], how="left")
if len(mbid_meta) > 0:
    merged = merged.merge(mbid_meta, on=["track_name", "artist_name"], how="left")
if len(acoustic_meta) > 0:
    merged = merged.merge(acoustic_meta, on=["track_name", "artist_name"], how="left")

print(f"✅ Merged dataset has {len(merged)} tracks")
print(f"   - Tracks with Last.fm data: {merged['lastfm_playcount'].notna().sum()}")
print(f"   - Tracks with MBID: {merged['mbid'].notna().sum()}")
print(f"   - Tracks with acoustic features: {merged['acoustic_danceability'].notna().sum()}")

# Save merged data
merged.to_csv(os.path.join(DATA_DIR, "merged_full.csv"), index=False)

# -----------------------------
# 5. Build viral labels
# -----------------------------
print("\n🏷️ Building viral labels based on Last.fm metrics...")


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
    release_date = pd.to_datetime(row.get("release_date"))

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


# Calculate labels
merged["viral_label"] = merged.apply(calculate_viral_label, axis=1)

# Add additional metrics for analysis
merged["listener_play_ratio"] = merged["lastfm_listeners"] / (merged["lastfm_playcount"] + 1e-9)
merged["days_since_release"] = (pd.Timestamp.now() - pd.to_datetime(merged["release_date"])).dt.days

# Save labeled data
merged.to_csv(os.path.join(DATA_DIR, "merged_labeled.csv"), index=False)

print(f"✅ Viral labels assigned: {merged['viral_label'].sum()} viral tracks out of {len(merged)}")
print(f"   Viral rate: {100 * merged['viral_label'].sum() / len(merged):.1f}%")

# Show distribution
print("\n📊 Label Distribution:")
print(merged['viral_label'].value_counts())

# -----------------------------
# 6. Train LightGBM classifier
# -----------------------------
print("\n🎯 Training LightGBM classifier...")

# Prepare training data
train_df = merged.dropna(subset=["viral_label"]).copy()
train_df["viral_label"] = train_df["viral_label"].astype(int)

# Define features
feature_cols = [
    "lastfm_listeners", "lastfm_playcount",
    "listener_play_ratio",  # Engagement metric
    "acoustic_danceability", "acoustic_tempo",
    "acoustic_energy", "acoustic_valence"
]
feature_cols = [f for f in feature_cols if f in train_df.columns]

# Remove rows with missing features
X = train_df[feature_cols].fillna(0)
y = train_df["viral_label"]

print(f"\n📈 Training Data Summary:")
print(f"   Total samples: {len(X)}")
print(f"   Features: {len(feature_cols)}")
print(f"   Positive samples (viral): {y.sum()}")
print(f"   Negative samples (not viral): {len(y) - y.sum()}")
print(f"   Class balance: {100 * y.sum() / len(y):.1f}% viral")

if len(y.unique()) < 2:
    print("\n⚠️ Not enough positive/negative samples to train model.")
    print("   Try lowering the thresholds (MIN_TOTAL_PLAYCOUNT, MIN_LISTENERS)")
    exit(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\n✂️ Train/Test Split:")
print(f"   Training: {len(X_train)} samples ({y_train.sum()} viral)")
print(f"   Testing: {len(X_test)} samples ({y_test.sum()} viral)")

# Train model
print("\n🚀 Training LightGBM model...")
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

# -----------------------------
# 7. Evaluate model
# -----------------------------
print("\n📊 Evaluating model...")

y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Metrics
auc = roc_auc_score(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

print(f"\n✅ Model Performance:")
print(f"   ROC-AUC Score: {auc:.4f}")
print(f"   Average Precision: {ap:.4f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Viral", "Viral"]))

# Feature importance
print("\n🎯 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:,.0f}")

# -----------------------------
# 8. Save model
# -----------------------------
model_path = os.path.join(DATA_DIR, "lgb_viral_predictor.pkl")
joblib.dump(model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save feature list
feature_list_path = os.path.join(DATA_DIR, "feature_list.txt")
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"💾 Feature list saved to: {feature_list_path}")

print("\n🏁 Pipeline complete! Model ready for predictions.")
print(f"\n📁 Output files in {DATA_DIR}:")
print(f"   - merged_full.csv: All merged data")
print(f"   - merged_labeled.csv: Data with viral labels")
print(f"   - lgb_viral_predictor.pkl: Trained model")
print(f"   - feature_list.txt: List of features")