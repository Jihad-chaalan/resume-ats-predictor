# tests/test_model_performance.py - UPDATED FOR V3.1
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    """Clean text the same way we did in training"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def test_model_performance():
    """
    This test runs the model on the test dataset.
    If performance drops below thresholds, the CI pipeline FAILS.
    """
    print("=" * 60)
    print("🔬 MODEL PERFORMANCE VALIDATION (v3.1 - with Squared Penalty)")
    print("=" * 60)
    
    # 1. Load the test data
    df = pd.read_csv('data/processed/cleaned_resume_data.csv')
    print(f"✅ Loaded {len(df)} rows")
    
    # 2. Prepare features (MUST MATCH TRAINING!)
    numeric_features = ['experience_years', 'job_experience_required', 'similarity_score', 'similarity_penalty']
    
    # Clean the resume text
    df['resume_cleaned'] = df['resume_text'].apply(clean_text)
    
    # Compute semantic similarity
    print("🔄 Computing semantic similarity for test data...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embeddings = embedder.encode(df['resume_cleaned'].tolist(), show_progress_bar=True, batch_size=32)
    jd_embeddings = embedder.encode(df['jd_cleaned'].tolist(), show_progress_bar=True, batch_size=32)
    
    similarities = []
    for i in range(len(df)):
        sim = cosine_similarity([resume_embeddings[i]], [jd_embeddings[i]])[0][0]
        similarities.append(sim)
    
    df['similarity_score'] = similarities
    df['similarity_penalty'] = (1 - np.array(similarities)) ** 2  # Squared penalty!
    print(f"✅ Added similarity_score and similarity_penalty (squared) to test data")
    
    X_text = df['resume_cleaned']
    X_num = df[numeric_features]
    y_true = df['shortlisted']
    
    def combine_features(text_series, num_df):
        df_combined = pd.DataFrame({'text': text_series})
        for col in num_df.columns:
            df_combined[col] = num_df[col].values
        return df_combined
    
    X_test = combine_features(X_text, X_num)
    
    # 3. Load the model
    model = joblib.load('models/model_xgboost.pkl')
    print("✅ Model loaded successfully")
    
    # 4. Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE ON TEST SET")
    print("=" * 60)
    print(f"🔹 Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"🔹 F1-Score:  {f1:.4f}")
    print(f"🔹 AUC-ROC:   {auc:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Rejected', 'Shortlisted']))
    
    MIN_ACCURACY = 0.75
    MIN_F1 = 0.75
    
    print("\n" + "=" * 60)
    print("🚦 PERFORMANCE GATEKEEPER")
    print("=" * 60)
    print(f"✅ Required Accuracy: >= {MIN_ACCURACY:.2f}  |  Actual: {accuracy:.4f}  |  {'✅ PASS' if accuracy >= MIN_ACCURACY else '❌ FAIL'}")
    print(f"✅ Required F1-Score: >= {MIN_F1:.2f}  |  Actual: {f1:.4f}  |  {'✅ PASS' if f1 >= MIN_F1 else '❌ FAIL'}")
    
    assert accuracy >= MIN_ACCURACY, f"❌ Accuracy {accuracy:.4f} is below threshold {MIN_ACCURACY:.2f}! Deployment blocked."
    assert f1 >= MIN_F1, f"❌ F1-Score {f1:.4f} is below threshold {MIN_F1:.2f}! Deployment blocked."
    
    print("\n" + "=" * 60)
    print("✅ ALL THRESHOLDS PASSED! Model is ready for deployment.")
    print("=" * 60)