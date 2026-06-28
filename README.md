# 📄 Resume ATS Predictor

[![Hugging Face Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Jihad-chaalan/resume-ats-predictor)
[![CI/CD](https://github.com/Jihad-chaalan/resume-ats-predictor/actions/workflows/deploy.yml/badge.svg)](https://github.com/Jihad-chaalan/resume-ats-predictor/actions)

> An end-to-end MLOps project that simulates how an Applicant Tracking System (ATS) screens resumes.  
> **Trained from scratch** on 6,000 resumes using XGBoost. Deployed with a fully automated CI/CD pipeline.

---

## 🚀 Live Demo

Test the model live on Hugging Face Spaces:  
👉 **[Click here to try it!](https://huggingface.co/spaces/jiha-d/resume-ats-predictor)**

---

## 📊 Model Performance (v2)

| Metric       | Score      |
| :----------- | :--------- |
| **Accuracy** | **85.75%** |
| **F1-Score** | **0.910**  |
| **AUC-ROC**  | **0.919**  |

### Top Predictive Features

| Feature                     | Importance    |
| :-------------------------- | :------------ |
| **experience_years**        | **0.0276** 🏆 |
| **job_experience_required** | 0.0165        |
| **similarity_penalty**      | 0.0045        |
| **similarity_score**        | 0.0040        |

---

## 🔧 Tech Stack

- **Language:** Python 3.11
- **Model:** XGBoost (trained from scratch)
- **MLOps:** GitHub Actions (CI/CD), Hugging Face Spaces, MLflow
- **UI:** Gradio
- **Libraries:** scikit-learn, pandas, numpy, joblib, sentence-transformers

---

## 🛠️ How It Works

1. **Text Preprocessing:** Resume text is cleaned (lowercase, punctuation removed, stopwords filtered).
2. **Feature Engineering:** TF-IDF extracts the top 5,000 word features, plus numeric features:
   - `experience_years` (candidate's experience)
   - `job_experience_required` (role requirement)
   - `similarity_score` (semantic relevance between resume and JD)
   - `similarity_penalty` (penalizes mismatched roles)
3. **Prediction:** The XGBoost model outputs a "Shortlist Probability" and a binary decision.
4. **Explainability:** The app shows **matched vs. missing keywords** to give actionable feedback to job seekers.

---

## ⚠️ Limitations & Learnings

### The Experience Bias

During testing, I discovered a critical limitation: **the model is heavily biased toward years of experience.**

| Test Case                               | Experience | Role Match    | Model Decision                        |
| :-------------------------------------- | :--------- | :------------ | :------------------------------------ |
| **Frontend Developer → Data Scientist** | 10 years   | ❌ Mismatched | ✅ **SHORTLISTED** (64.2% confidence) |
| **Frontend Developer → Data Scientist** | 2 years    | ❌ Mismatched | ❌ **REJECTED** (91.5% confidence)    |

### Why This Happens

Looking at the feature importance:

| Feature              | Importance | Impact                       |
| :------------------- | :--------- | :--------------------------- |
| `experience_years`   | **0.0276** | 🏆 Strongest predictor       |
| `similarity_penalty` | **0.0045** | ⚠️ 6x weaker than experience |

Because `experience_years` is **6 times stronger** than the similarity penalty, the model overrides role relevance when a candidate has high experience. A Frontend Developer with 10+ years gets shortlisted for a Data Scientist role because the dataset prioritized tenure over role matching.

### What I Learned

1. **Synthetic datasets carry hidden biases.** The generated data had a rule: _"If experience matches, mark them as shortlisted."_ The model learned this rule perfectly.
2. **Feature engineering is not just about adding features—it's about ensuring they have enough influence.** The similarity penalty was too weak to counteract the experience bias.
3. **Testing edge cases reveals model flaws.** Manually testing mismatched roles uncovered this bias that wouldn't have been obvious from accuracy metrics alone.

### How This Would Be Fixed in Production

| Approach                | Description                                                                 |
| :---------------------- | :-------------------------------------------------------------------------- |
| **Stronger Penalty**    | Multiply `similarity_penalty` by 10x to enforce role relevance.             |
| **Business Logic Rule** | Auto-reject any candidate with similarity < 0.30, regardless of experience. |
| **Balanced Dataset**    | Train on more examples of high-experience mismatched roles.                 |

---
