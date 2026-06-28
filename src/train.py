# src/train.py - XGBOOST WITH SEMANTIC SIMILARITY + SQUARED PENALTY + MLflow
# Version: v3.1 - Final Production

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import joblib
import mlflow
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. SET UP MLFLOW VERSIONING
# ============================================================
mlflow.set_experiment("Resume ATS Predictor")
run_name = f"v3.1_squared_penalty_{datetime.now().strftime('%Y%m%d_%H%M')}"

print("=" * 60)
print("RESUME ATS PREDICTOR - XGBOOST TRAINING (v3.1 - Squared Penalty)")
print("=" * 60)
print(f"🔬 MLflow Run: {run_name}")

# ============================================================
# 2. LOAD DATA
# ============================================================
df = pd.read_csv('data/processed/cleaned_resume_data.csv')
print(f"✅ Loaded {len(df)} rows")

# ============================================================
# 3. COMPUTE SEMANTIC SIMILARITY + SQUARED PENALTY
# ============================================================
print("\n🚀 Computing semantic similarity...")
print("   (Using all-MiniLM-L6-v2 - CPU-friendly, ~80MB)")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("📊 Encoding resumes...")
resume_embeddings = embedder.encode(
    df['resume_cleaned'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

print("📊 Encoding job descriptions...")
jd_embeddings = embedder.encode(
    df['jd_cleaned'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

print("📊 Calculating cosine similarity...")
similarities = []
for i in range(len(df)):
    sim = cosine_similarity([resume_embeddings[i]], [jd_embeddings[i]])[0][0]
    similarities.append(sim)

# Convert to numpy array
similarities = np.array(similarities)

# Add features: similarity_score and squared_penalty (more aggressive)
df['similarity_score'] = similarities
df['similarity_penalty'] = (1 - similarities) ** 2  # <-- Squared penalty!

print(f"\n✅ Added 'similarity_score' and 'similarity_penalty' (squared)!")
print(f"   📊 Mean similarity: {np.mean(similarities):.3f}")
print(f"   📊 Mean penalty: {np.mean(df['similarity_penalty']):.3f}")
print(f"   📊 Min similarity: {np.min(similarities):.3f}")
print(f"   📊 Max similarity: {np.max(similarities):.3f}")

# ============================================================
# 4. DEFINE FEATURES (4 NUMERIC FEATURES)
# ============================================================
numeric_features = [
    'experience_years',
    'job_experience_required',
    'similarity_score',
    'similarity_penalty'
]
text_feature = 'resume_cleaned'

X_text = df[text_feature]
X_num = df[numeric_features]
y = df['shortlisted']

print(f"\n✅ Features: Text + Numeric ({numeric_features})")
print(f"✅ Target distribution: {y.value_counts().to_dict()}")

# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train size: {len(X_text_train)}, Test size: {len(X_text_test)}")

# ============================================================
# 6. PREPARE DATA FOR PIPELINE
# ============================================================
def combine_features(text_series, num_df):
    """Combine text and numeric features into one DataFrame"""
    df_combined = pd.DataFrame({'text': text_series})
    for col in num_df.columns:
        df_combined[col] = num_df[col].values
    return df_combined

X_train_combined = combine_features(X_text_train, X_num_train)
X_test_combined = combine_features(X_text_test, X_num_test)

print(f"✅ Combined features shape: {X_train_combined.shape}")

# ============================================================
# 7. BUILD PIPELINE
# ============================================================
print("\n🔧 Building pipeline...")

# Text pipeline: TF-IDF with bigrams
text_pipeline = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)  # Capture "machine learning" as one feature
)

# Numeric pipeline: Standard scaling
num_pipeline = StandardScaler()

# Combine pipelines
preprocessor = ColumnTransformer([
    ('text', text_pipeline, 'text'),
    ('num', num_pipeline, numeric_features)
])

# XGBoost Classifier with scale_pos_weight=2 to handle imbalance
classifier = xgb.XGBClassifier(
    n_estimators=120,          # Slightly more trees
    max_depth=7,               # Slightly deeper trees
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2,        # Give more weight to minority class (Rejected)
    random_state=42,
    eval_metric='logloss'
)

# Full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

print("✅ Pipeline built successfully!")

# ============================================================
# 8. TRAIN WITH MLFLOW TRACKING
# ============================================================
with mlflow.start_run(run_name=run_name) as run:
    # Log ALL parameters to MLflow
    mlflow.log_params({
        "model_type": "XGBoost",
        "version": "v3.1",
        "improvement": "squared_similarity_penalty",
        "max_features": 5000,
        "n_estimators": 120,
        "max_depth": 7,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 2,
        "numeric_features": str(numeric_features),
        "text_feature": text_feature,
        "train_size": len(X_text_train),
        "test_size": len(X_text_test),
        "embedding_model": "all-MiniLM-L6-v2",
        "penalty_type": "squared"
    })

    print("\n🚀 Training XGBoost from scratch...")
    model.fit(X_train_combined, y_train)
    print("✅ Training complete!")

    # ============================================================
    # 9. EVALUATE MODEL
    # ============================================================
    y_pred = model.predict(X_test_combined)
    y_proba = model.predict_proba(X_test_combined)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc
    })

    # Print results
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Shortlisted']))

    # ============================================================
    # 10. FEATURE IMPORTANCE
    # ============================================================
    print("\n" + "=" * 60)
    print("🏆 TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)

    text_feature_names = model.named_steps['preprocessor'].named_transformers_['text'].get_feature_names_out().tolist()
    numeric_feature_names = numeric_features
    all_feature_names = text_feature_names + numeric_feature_names

    importance = model.named_steps['classifier'].feature_importances_
    top_indices = np.argsort(importance)[-10:][::-1]

    print("\nTop 10 features that most strongly predict SHORTLISTING:")
    for i, idx in enumerate(top_indices, 1):
        feature_name = all_feature_names[idx]
        if feature_name in numeric_features:
            print(f"  {i}. 📊 {feature_name}: {importance[idx]:.4f} (NUMERIC)")
        else:
            print(f"  {i}. 📝 '{feature_name}': {importance[idx]:.4f}")

    # Check if similarity_penalty is in the top 10
    penalty_rank = None
    for i, idx in enumerate(top_indices, 1):
        feature_name = all_feature_names[idx]
        if feature_name == 'similarity_penalty':
            penalty_rank = i
            break

    if penalty_rank:
        print(f"\n✅ similarity_penalty is in the top {penalty_rank} features!")
        print("   (This means the model will heavily penalize mismatched roles like Laravel → Data Science)")
    else:
        print("\n⚠️ similarity_penalty is NOT in the top 10 features.")
        print("   (You may need to increase its importance further)")

    # ============================================================
    # 11. SAVE MODEL LOCALLY
    # ============================================================
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, 'models/model_xgboost.pkl')
    print("\n✅ Model saved locally to: models/model_xgboost.pkl")

    # ============================================================
    # 12. LOG MODEL TO MLFLOW
    # ============================================================
    mlflow.log_artifact("models/model_xgboost.pkl")
    print("✅ Model logged to MLflow as an artifact")

    # ============================================================
    # 13. SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("✅ MLFLOW VERSIONING SUMMARY")
    print("=" * 60)
    print(f"📊 Run ID: {run.info.run_id}")
    print(f"📊 Experiment: Resume ATS Predictor")
    print(f"📊 Run Name: {run_name}")
    print(f"📊 View with: mlflow ui")
    print("\n✅ Model is ready for deployment!")
    print("🚀 Run 'python app.py' to test locally")
    print("   or push to GitHub for CI/CD deployment")