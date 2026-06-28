# src/train.py - XGBOOST VERSION (FIXED)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from pathlib import Path

print("=" * 50)
print("RESUME ATS PREDICTOR - XGBOOST TRAINING")
print("=" * 50)

# 1. Load the cleaned data
df = pd.read_csv('data/processed/cleaned_resume_data.csv')
print(f"✅ Loaded {len(df)} rows")

# 2. Define Features and Target
numeric_features = ['experience_years', 'job_experience_required']
text_feature = 'resume_cleaned'

X_text = df[text_feature]
X_num = df[numeric_features]
y = df['shortlisted']

print(f"✅ Features: Text + Numeric ({numeric_features})")
print(f"✅ Target distribution: {y.value_counts().to_dict()}")

# 3. Split the data
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train size: {len(X_text_train)}, Test size: {len(X_text_test)}")

# 4. Combine text and numeric features into one DataFrame for the pipeline
def combine_features(text_series, num_df):
    df_combined = pd.DataFrame({'text': text_series})
    for col in num_df.columns:
        df_combined[col] = num_df[col].values
    return df_combined

X_train_combined = combine_features(X_text_train, X_num_train)
X_test_combined = combine_features(X_text_test, X_num_test)

# 5. Build the XGBoost Pipeline
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

# 6. Train the model (FROM SCRATCH!)
print("\n🚀 Training XGBoost from scratch...")
model.fit(X_train_combined, y_train)
print("✅ Training complete!")

# 7. Evaluate
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

# 8. Feature Importance (FIXED)
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

# 9. Save the model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, 'models/model_xgboost.pkl')
print("\n✅ Model saved to: models/model_xgboost.pkl")
print("🚀 Ready for deployment!")