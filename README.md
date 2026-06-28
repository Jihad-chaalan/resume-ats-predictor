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

## 📊 Model Performance

| Metric       | Score      |
| :----------- | :--------- |
| **Accuracy** | **86.25%** |
| **F1-Score** | **0.911**  |
| **AUC-ROC**  | **0.906**  |

### Top Predictive Features

1. **Experience Years** (Strongest predictor)
2. **Required Experience** (Job requirement)
3. Keywords like _certifications, experience, candidate, certified_

---

## 🔧 Tech Stack

- **Language:** Python 3.11
- **Model:** XGBoost (trained from scratch)
- **MLOps:** GitHub Actions (CI/CD), Hugging Face Spaces
- **UI:** Gradio
- **Libraries:** scikit-learn, pandas, numpy, joblib

---

## 🛠️ How It Works

1. **Text Preprocessing:** Resume text is cleaned (lowercase, punctuation removed, stopwords filtered).
2. **Feature Engineering:** TF-IDF extracts the top 5,000 word features, plus numeric features (_Years of Experience_, _Required Experience_).
3. **Prediction:** The XGBoost model outputs a "Shortlist Probability" and a binary decision.
4. **Explainability:** The app shows **matched vs. missing keywords** to give actionable feedback to job seekers.

---
