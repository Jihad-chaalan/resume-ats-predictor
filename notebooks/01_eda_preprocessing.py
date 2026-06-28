# %% [markdown]
# # Resume ATS Predictor - EDA & Preprocessing
# ## NO SEABORN VERSION (Works on Python 3.12)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# Use built-in matplotlib style (no seaborn needed)
plt.style.use('ggplot')
print("✅ Libraries loaded!")

# %%
# Load dataset
data_dir = Path("../data/raw")
csv_files = list(data_dir.glob("*.csv"))

if csv_files:
    file_path = csv_files[0]
    print(f"📂 Loading: {file_path.name}")
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError("No CSV file found in data/raw/")

print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Column names: {df.columns.tolist()}")

# %%
# Target distribution
target_counts = df['shortlisted'].value_counts()
print(f"Shortlisted: {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"Not Shortlisted: {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")

plt.figure(figsize=(6, 4))
plt.bar(['Rejected', 'Shortlisted'], [target_counts.get(0, 0), target_counts.get(1, 0)], 
        color=['#ff6b6b', '#51cf66'])
plt.title('Class Distribution', fontsize=14)
plt.ylabel('Count')
plt.show()

# %%
# Text length analysis
df['resume_length'] = df['resume_text'].str.len()
df['jd_length'] = df['job_description'].str.len()
print(f"Resume - Min: {df['resume_length'].min():,}, Max: {df['resume_length'].max():,}, Mean: {df['resume_length'].mean():,.0f}")
print(f"JD - Min: {df['jd_length'].min():,}, Max: {df['jd_length'].max():,}, Mean: {df['jd_length'].mean():,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['resume_length'], bins=30, color='#4dabf7', edgecolor='black', alpha=0.7)
axes[0].set_title('Resume Length Distribution')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Count')

axes[1].hist(df['jd_length'], bins=30, color='#fcc419', edgecolor='black', alpha=0.7)
axes[1].set_title('Job Description Length Distribution')
axes[1].set_xlabel('Characters')
plt.tight_layout()
plt.show()

# %%
# Check numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Available numeric columns: {numeric_cols}")

for col in numeric_cols:
    if col not in ['shortlisted', 'resume_length', 'jd_length']:
        corr = df[col].corr(df['shortlisted'])
        print(f"{col}: correlation with target = {corr:.3f}")

# %%
# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test the function
sample = df['resume_text'].iloc[0]
print(f"Original: {sample[:100]}...")
print(f"Cleaned:  {clean_text(sample)[:100]}...")

df['resume_cleaned'] = df['resume_text'].apply(clean_text)
df['jd_cleaned'] = df['job_description'].apply(clean_text)
print("✅ Text cleaned")

# %%
# Save processed data
Path("../data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv('../data/processed/cleaned_resume_data.csv', index=False)
print("✅ Saved to: ../data/processed/cleaned_resume_data.csv")

# %%
print("\n" + "="*50)
print("EDA COMPLETE - SUMMARY")
print("="*50)
print(f"✅ Total samples: {len(df):,}")
print(f"✅ Features: {len(df.columns)}")
print(f"✅ Shortlisted: {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"✅ Rejected: {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
print("\n🚀 Ready for Feature Engineering & Model Training!")
# %%
