# app.py - Resume ATS Predictor Web App
import gradio as gr
import joblib
import pandas as pd
import re
from pathlib import Path

# Load the trained model
model = joblib.load('models/model_xgboost.pkl')

def clean_text(text):
    """Clean text the same way we did in training"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_keyword_gap(resume_text, job_description):
    # Clean punctuation from words
    resume_words = set(re.findall(r'\b[a-zA-Z]+\b', resume_text.lower()))
    jd_words = set(re.findall(r'\b[a-zA-Z]+\b', job_description.lower()))
    
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'for', 'on', 'at', 'to', 'in', 'with', 'without', 'of', 'by', 'we', 'is', 'are', 'am'}
    jd_words = jd_words - stopwords
    resume_words = resume_words - stopwords
    
    missing = jd_words - resume_words
    matched = jd_words & resume_words
    return list(missing)[:10], list(matched)[:10]

def predict_resume(resume_text, job_description, years_experience, job_experience_required):
    """
    Main prediction function for the Gradio app
    """
    if not resume_text or not job_description:
        return "Please provide both resume and job description.", "", "", "", ""
    
    try:
        # Clean the resume text
        cleaned_resume = clean_text(resume_text)
        
        # Create input DataFrame (matching training format)
        input_data = pd.DataFrame({
            'text': [cleaned_resume],
            'experience_years': [float(years_experience) if years_experience else 0],
            'job_experience_required': [float(job_experience_required) if job_experience_required else 0]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Get keyword gap
        missing_keywords, matched_keywords = get_keyword_gap(resume_text, job_description)
        
        # Prepare results
        if prediction == 1:
            decision = "✅ SHORTLISTED"
            score_color = "green"
            confidence = probability[1]
        else:
            decision = "❌ REJECTED"
            score_color = "red"
            confidence = probability[0]
        
        # Format missing keywords
        missing_str = ", ".join(missing_keywords) if missing_keywords else "None! Great match!"
        matched_str = ", ".join(matched_keywords[:10]) if matched_keywords else "No keywords found"
        
        return (
            decision,
            f"Confidence: {confidence:.1%}",
            f"📊 Match Score: {probability[1]:.1%}",
            f"✅ Found keywords: {matched_str}",
            f"❌ Missing keywords: {missing_str}"
        )
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", ""

# Create the Gradio interface
with gr.Blocks(title="Resume ATS Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 Resume ATS Predictor
    ### Predict if your resume will be shortlisted by an Applicant Tracking System
    
    Enter your resume text, the job description, and your experience details below.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            resume_input = gr.Textbox(
                label="📝 Resume Text",
                placeholder="Paste your resume text here...",
                lines=10
            )
            jd_input = gr.Textbox(
                label="💼 Job Description",
                placeholder="Paste the job description here...",
                lines=8
            )
            
            with gr.Row():
                exp_input = gr.Number(
                    label="📅 Your Years of Experience",
                    value=0,
                    precision=1
                )
                jd_exp_input = gr.Number(
                    label="📅 Required Years of Experience",
                    value=0,
                    precision=1
                )
            
            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            decision_output = gr.Textbox(
                label="Decision",
                lines=1,
                interactive=False
            )
            confidence_output = gr.Textbox(
                label="Confidence",
                lines=1,
                interactive=False
            )
            score_output = gr.Textbox(
                label="Match Score",
                lines=1,
                interactive=False
            )
            matched_output = gr.Textbox(
                label="✅ Keywords Found",
                lines=3,
                interactive=False
            )
            missing_output = gr.Textbox(
                label="❌ Missing Keywords",
                lines=3,
                interactive=False
            )
    
    # Connect the button to the prediction function
    predict_btn.click(
        fn=predict_resume,
        inputs=[resume_input, jd_input, exp_input, jd_exp_input],
        outputs=[decision_output, confidence_output, score_output, matched_output, missing_output]
    )
    
    gr.Markdown("""
    ---
    ### 💡 How it works
    This model uses **XGBoost** trained on 6,000 resumes with TF-IDF text features.
    It predicts shortlisting probability based on:
    - Your resume content (keywords, skills, experience descriptions)
    - Your years of experience vs. job requirements
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)