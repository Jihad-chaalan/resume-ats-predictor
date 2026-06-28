# src/train.py - XGBOOST WITH SEMANTIC SIMILARITY (IMPROVEMENT 1)
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
from pathlib import Path

print("=" * 50)
print("RESUME ATS PREDICTOR - XGBOOST TRAINING (v2 - with Similarity)")
print("=" * 50)

# 1. Load the cleaned data
df = pd.read_csv('data/processed/cleaned_resume_data.csv')
print(f"✅ Loaded {len(df)} rows")

# ============================================================
# 2. ADD SEMANTIC SIMILARITY FEATURE (THE LARAVEL FIX!)
# ============================================================
print("\n🚀 Computing semantic similarity between resumes and job descriptions...")
print("   (This uses a tiny, CPU-friendly AI model: all-MiniLM-L6-v2)")

# Load the sentence transformer (lightweight, ~80MB, runs on CPU)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings (numerical representations) for cleaned text
resume_embeddings = embedder.encode(
    df['resume_cleaned'].tolist(), 
    show_progress_bar=True,
    batch_size=32
)
jd_embeddings = embedder.encode(
    df['jd_cleaned'].tolist(), 
    show_progress_bar=True,
    batch_size=32
)

# Calculate cosine similarity between each resume and its corresponding JD
similarities = []
for i in range(len(df)):
    sim = cosine_similarity([resume_embeddings[i]], [jd_embeddings[i]])[0][0]
    similarities.append(sim)

df['similarity_score'] = similarities
print(f"✅ Added 'similarity_score' feature!")
print(f"   📊 Mean similarity: {np.mean(similarities):.3f}")
print(f"   📊 Min similarity: {np.min(similarities):.3f}")
print(f"   📊 Max similarity: {np.max(similarities):.3f}")

# ============================================================
# 3. Define Features and Target (NOW WITH SIMILARITY!)
# ============================================================
numeric_features = ['experience_years', 'job_experience_required', 'similarity_score']
text_feature = 'resume_cleaned'

X_text = df[text_feature]
X_num = df[numeric_features]
y = df['shortlisted']

print(f"\n✅ Features: Text + Numeric ({numeric_features})")
print(f"✅ Target distribution: {y.value_counts().to_dict()}")

# 4. Split the data
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train size: {len(X_text_train)}, Test size: {len(X_text_test)}")

# 5. Combine text and numeric features into one DataFrame for the pipeline
def combine_features(text_series, num_df):
    df_combined = pd.DataFrame({'text': text_series})
    for col in num_df.columns:
        df_combined[col] = num_df[col].values
    return df_combined

X_train_combined = combine_features(X_text_train, X_num_train)
X_test_combined = combine_features(X_text_test, X_num_test)

# 6. Build the XGBoost Pipeline
text_pipeline = TfidfVectorizer(max_features=5000, stop_words='english')
num_pipeline = StandardScaler()

preprocessor = ColumnTransformer([
    ('text', text_pipeline, 'text'),
    ('num', num_pipeline, numeric_features)
])

# XGBoost Classifier - trained from scratch
classifier = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='logloss'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# 7. Train the model (FROM SCRATCH!)
print("\n🚀 Training XGBoost from scratch...")
model.fit(X_train_combined, y_train)
print("✅ Training complete!")

# 8. Evaluate
y_pred = model.predict(X_test_combined)
y_proba = model.predict_proba(X_test_combined)[:, 1]

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Shortlisted']))

# 9. Feature Importance (FIXED)
print("\n" + "=" * 50)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 50)

text_feature_names = model.named_steps['preprocessor'].named_transformers_['text'].get_feature_names_out().tolist()
numeric_feature_names = numeric_features
all_feature_names = text_feature_names + numeric_feature_names

importance = model.named_steps['classifier'].feature_importances_
top_indices = np.argsort(importance)[-10:][::-1]

print("\nTop 10 features that most strongly predict SHORTLISTING:")
for idx in top_indices:
    feature_name = all_feature_names[idx]
    if feature_name in numeric_features:
        print(f"  📊 {feature_name}: {importance[idx]:.4f} (NUMERIC)")
    else:
        print(f"  📝 '{feature_name}': {importance[idx]:.4f}")

# 10. Save the model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, 'models/model_xgboost.pkl')
print("\n✅ Model saved to: models/model_xgboost.pkl")
print("🚀 Ready for deployment!")