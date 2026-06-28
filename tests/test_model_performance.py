# tests/test_model_performance.py
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

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
    print("🔬 MODEL PERFORMANCE VALIDATION")
    print("=" * 60)
    
    # 1. Load the test data
    df = pd.read_csv('data/processed/cleaned_resume_data.csv')
    print(f"✅ Loaded {len(df)} rows")
    
    # 2. Prepare features (same as training)
    numeric_features = ['experience_years', 'job_experience_required']
    
    # Clean the resume text
    df['resume_cleaned'] = df['resume_text'].apply(clean_text)
    
    X_text = df['resume_cleaned']
    X_num = df[numeric_features]
    y_true = df['shortlisted']
    
    # Combine features the same way the pipeline expects
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
    
    # 5. Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE ON TEST SET")
    print("=" * 60)
    print(f"🔹 Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"🔹 F1-Score:  {f1:.4f}")
    print(f"🔹 AUC-ROC:   {auc:.4f}")
    
    # Print the classification report for full transparency
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Rejected', 'Shortlisted']))
    
    # 6. Define thresholds (The "Gatekeeper")
    MIN_ACCURACY = 0.80
    MIN_F1 = 0.80
    
    print("\n" + "=" * 60)
    print("🚦 PERFORMANCE GATEKEEPER")
    print("=" * 60)
    print(f"✅ Required Accuracy: >= {MIN_ACCURACY:.2f}  |  Actual: {accuracy:.4f}  |  {'✅ PASS' if accuracy >= MIN_ACCURACY else '❌ FAIL'}")
    print(f"✅ Required F1-Score: >= {MIN_F1:.2f}  |  Actual: {f1:.4f}  |  {'✅ PASS' if f1 >= MIN_F1 else '❌ FAIL'}")
    
    # 7. Fail the test if thresholds are not met
    assert accuracy >= MIN_ACCURACY, f"❌ Accuracy {accuracy:.4f} is below threshold {MIN_ACCURACY:.2f}! Deployment blocked."
    assert f1 >= MIN_F1, f"❌ F1-Score {f1:.4f} is below threshold {MIN_F1:.2f}! Deployment blocked."
    
    print("\n" + "=" * 60)
    print("✅ ALL THRESHOLDS PASSED! Model is ready for deployment.")
    print("=" * 60)
    
