from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
from resumes_parser import extract_resume_details
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from ats_checker import check_ats_compatibility, extract_text_from_pdf, extract_text_from_docx, benchmark_against_industry, check_file_format
import tempfile

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)
# Enable CORS for all routes to work with React
CORS(app)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_phone_number(text):
    """Extract phone number from text using regex patterns"""
    if not text:
        return "Not Found"

    # Define various phone number patterns
    patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{4}',  # International format
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format (xxx) xxx-xxxx
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # Simple format xxx-xxx-xxxx
        r'\d{10}',  # Just 10 digits
        r'\+?\d{1,3}[-.\s]?\d{9,10}'  # International with country code
    ]

    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]

    return "Not Found"

def cleanResume(txt):
    """Enhanced text cleaning function with lemmatization and improved regex"""
    if not txt:
        return ""

    # If txt is a list, convert it to string first
    if isinstance(txt, list):
        txt = " ".join(str(item) for item in txt)

    # Convert to lowercase
    cleanText = txt.lower()

    # Remove URLs, email addresses, and social media handles
    cleanText = re.sub(r'http\S+', ' ', cleanText)
    cleanText = re.sub(r'www\.\S+', ' ', cleanText)
    cleanText = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)

    # Remove special characters and numbers
    cleanText = re.sub(r'[^\w\s]', ' ', cleanText)
    cleanText = re.sub(r'\d+', ' ', cleanText)

    # Remove extra whitespaces
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()

    # Simple word tokenization without using nltk.word_tokenize
    words = cleanText.split()

    # Filter and lemmatize words
    filtered_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            try:
                lemma = lemmatizer.lemmatize(word)
                filtered_words.append(lemma)
            except:
                filtered_words.append(word)

    return ' '.join(filtered_words)

def extract_key_features(resume_data):
    """Extract and weight important sections from resume data"""
    features = {}

    # Get raw text as base
    raw_text = resume_data.get("Raw Text", "")

    # Convert skills to string regardless of type
    skills = resume_data.get("Skills", [])
    if isinstance(skills, list):
        features['skills'] = " ".join(str(skill) for skill in skills)
    else:
        features['skills'] = str(skills)

    # Extract other sections with type checking
    for key in ["Education", "Projects", "Certifications"]:
        value = resume_data.get(key, "")
        if isinstance(value, list):
            features[key.lower()] = " ".join(str(item) for item in value)
        else:
            features[key.lower()] = str(value)

    # Handle Experience (now a list of dictionaries)
    experience = resume_data.get("Experience", [])
    if isinstance(experience, list) and experience and experience != ["Not Found"]:
        # Extract relevant text from each experience entry
        experience_text = []
        for exp in experience:
            if isinstance(exp, dict):
                # Combine Job Title, Company, and Key Responsibilities
                exp_text = f"{exp.get('Job Title', '')} {exp.get('Company', '')}"
                responsibilities = exp.get('Key Responsibilities', [])
                if responsibilities:
                    exp_text += " " + " ".join(responsibilities)
                experience_text.append(exp_text)
        features['experience'] = " ".join(experience_text)
    else:
        features['experience'] = ""

    # Clean each feature with error handling
    for key in features:
        try:
            features[key] = cleanResume(features[key])
        except Exception as e:
            print(f"Warning: Error cleaning {key}: {str(e)}")
            features[key] = ""

    # Create weighted feature text with explicit string concatenation
    weighted_text = ""

    # Add skills with triple weight
    weighted_text += (features['skills'] + " ") * 3

    # Add experience with double weight
    weighted_text += (features['experience'] + " ") * 2

    # Add other sections
    weighted_text += features['education'] + " "
    weighted_text += features['projects'] + " " if 'projects' in features else ""
    weighted_text += features['certifications'] + " " if 'certifications' in features else ""

    # Add cleaned raw text if weighted text is too short
    if len(weighted_text.strip()) < 100:
        try:
            if isinstance(raw_text, list):
                raw_text = " ".join(str(item) for item in raw_text)
            cleaned_raw = cleanResume(raw_text)
            weighted_text += " " + cleaned_raw
        except Exception as e:
            print(f"Warning: Error cleaning raw text: {str(e)}")

    return weighted_text.strip()

# Base directory - updated to match your actual path
BASE_PATH = r"C:/Users/yaswa/OneDrive/Desktop/JOBLENS/model"

# Define model paths
MODEL_PATHS = {
    "categorization": {
        "nb": os.path.join(BASE_PATH, "categorization/naive_bayes_categorization_model.pkl"),
        "logistic": os.path.join(BASE_PATH, "categorization/logistic_categorization_model.pkl"),
        "knn": os.path.join(BASE_PATH, "categorization/knn_categorization.pkl"),
        "label_encoder": os.path.join(BASE_PATH, "categorization/label_encoder_categorization.pkl"),
        "tfidf": os.path.join(BASE_PATH, "tfidf_shared/tfidf_vectorizer_categorization.pkl")
    },
    "recommendation": {
        "nb": os.path.join(BASE_PATH, "recommendation/naive_bayes_recommendation_model.pkl"),
        "logistic": os.path.join(BASE_PATH, "recommendation/logistic_recommendation_model.pkl"),
        "knn": os.path.join(BASE_PATH, "recommendation/knn_recommendation.pkl"),
        "label_encoder": os.path.join(BASE_PATH, "recommendation/label_encoder_job.pkl"),
        "tfidf": os.path.join(BASE_PATH, "tfidf_shared/tfidf_vectorizer_job_recommendation.pkl")
    }
}

def load_model(path):
    try:
        if not os.path.exists(path):
            print(f"❌ ERROR: Path does not exist: {path}")
            return None
        model = joblib.load(path)
        print(f"✅ Successfully loaded model from: {path}")
        return model
    except Exception as e:
        print(f"❌ ERROR: Failed to load model from {path}: {str(e)}")
        return None

# Load all models at startup
print("Loading models...")
models = {
    category: {model: load_model(path) for model, path in paths.items()}
    for category, paths in MODEL_PATHS.items()
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    temp_path = None
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]
        if not file.filename.endswith((".pdf", ".docx")):
            return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

        # Create a temporary file with a unique name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_path = tmp.name
            file.save(temp_path)

        print(f"Processing file: {file.filename} (saved as {temp_path})")

        # Extract resume details
        resume_data = extract_resume_details(temp_path)

        if not resume_data or "error" in resume_data:
            return jsonify({"error": "Failed to extract resume details"}), 400

        # Print the entire resume_data to debug
        print("Resume data keys:", resume_data.keys())

        # Extract display information with type checking
        name = str(resume_data.get("Name", "Not Found"))
        email = str(resume_data.get("Email", "Not Found"))

        # Try to extract phone number with different approaches
        phone = resume_data.get("Contact Number", None)

        # If phone is None or "Not Found", try to extract from raw text
        if not phone or phone == "Not Found" or phone == "None":
            raw_text = resume_data.get("Raw Text", "")
            if raw_text:
                phone = extract_phone_number(raw_text)
                print(f"Extracted phone from raw text: {phone}")
        else:
            phone = str(phone)

        print(f"Final phone number: {phone}")

        # Handle skills section properly
        skills = resume_data.get("Skills", [])
        if isinstance(skills, list):
            skills_display = ", ".join(str(skill) for skill in skills)
        else:
            skills_display = str(skills)

        # Get education
        education = resume_data.get("Education", "Not Found")
        if isinstance(education, list):
            education = "\n".join(str(edu) for edu in education)

        # Get experience (keep as a list of dictionaries)
        experience = resume_data.get("Experience", "Not Found")

        # Extract enhanced features from the resume
        try:
            feature_text = extract_key_features(resume_data)
            print(f"Extracted feature text length: {len(feature_text)} characters")
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            # Fallback to raw text with simple cleaning
            raw_text = resume_data.get("Raw Text", "")
            if isinstance(raw_text, list):
                raw_text = " ".join(str(item) for item in raw_text)
            feature_text = re.sub(r'[^\w\s]', ' ', raw_text.lower())
            feature_text = re.sub(r'\s+', ' ', feature_text).strip()
            print("Using fallback text cleaning")

        if not feature_text:
            return jsonify({"error": "Failed to extract meaningful features from resume"}), 400

        # Get user-selected models
        # Default to one of the available models, e.g., 'knn'
        categorization_model = request.form.get("categorization_model", "knn")
        recommendation_model = request.form.get("recommendation_model", "knn")

        print(f"Selected Models - Categorization: {categorization_model}, Recommendation: {recommendation_model}")

        if categorization_model not in models["categorization"] or recommendation_model not in models["recommendation"]:
            return jsonify({"error": "Invalid model selection"}), 400

        # Load TF-IDF vectorizers
        cat_tfidf = models["categorization"].get("tfidf")
        rec_tfidf = models["recommendation"].get("tfidf")

        if not cat_tfidf or not rec_tfidf:
            return jsonify({"error": "TF-IDF vectorizer could not be loaded. Check the server logs for details on the missing file."}), 500

        # Transform features using TF-IDF
        input_tfidf_cat = cat_tfidf.transform([feature_text])
        input_tfidf_rec = rec_tfidf.transform([feature_text])

        # Get prediction probabilities when available
        cat_model = models["categorization"].get(categorization_model)
        rec_model = models["recommendation"].get(recommendation_model)

        if not cat_model or not rec_model:
            return jsonify({"error": "Selected model could not be loaded"}), 500

        # Make predictions
        category_pred = int(cat_model.predict(input_tfidf_cat)[0])
        job_pred = int(rec_model.predict(input_tfidf_rec)[0])

        # Get prediction probabilities if model supports it
        cat_probs = {}
        job_probs = {}

        try:
            if hasattr(cat_model, 'predict_proba'):
                cat_prob_values = cat_model.predict_proba(input_tfidf_cat)[0]
                category_encoder = models["categorization"].get("label_encoder")
                cat_classes = category_encoder.classes_
                cat_probs = {str(category_encoder.inverse_transform([i])[0]): round(float(prob)*100, 2)
                            for i, prob in enumerate(cat_prob_values) if prob > 0.05}  # Show only probabilities > 5%
        except Exception as e:
            print(f"Warning: Could not get category probabilities: {str(e)}")

        try:
            if hasattr(rec_model, 'predict_proba'):
                job_prob_values = rec_model.predict_proba(input_tfidf_rec)[0]
                job_encoder = models["recommendation"].get("label_encoder")
                job_classes = job_encoder.classes_
                job_probs = {str(job_encoder.inverse_transform([i])[0]): round(float(prob)*100, 2)
                            for i, prob in enumerate(job_prob_values) if prob > 0.05}  # Show only probabilities > 5%
        except Exception as e:
            print(f"Warning: Could not get job recommendation probabilities: {str(e)}")

        print(f"Raw Predictions - Category: {category_pred}, Job: {job_pred}")

        # Load label encoders
        category_encoder = models["categorization"].get("label_encoder")
        job_encoder = models["recommendation"].get("label_encoder")

        category_label = str(category_encoder.inverse_transform([category_pred])[0]) if category_encoder else "Unknown"
        job_label = str(job_encoder.inverse_transform([job_pred])[0]) if job_encoder else "Unknown"

        print(f"Decoded Predictions - Category: {category_label}, Job: {job_label}")

        # Prepare response
        response = {
            "Resume Data": {
                "Name": name,
                "Email": email,
                "Phone": phone,
                "Skills": skills_display,
                "Education": education,
                "Experience": experience  # Keep as a list of dictionaries
            },
            "Predictions": {
                "Categorization": category_label,
                "Job Recommendation": job_label
            }
        }

        # Add probabilities to response if available
        if cat_probs:
            response["Predictions"]["Category Probabilities"] = cat_probs
        if job_probs:
            response["Predictions"]["Job Recommendation Probabilities"] = job_probs

        return jsonify(response)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Temporary file {temp_path} removed successfully")
            except Exception as e:
                print(f"Error removing temporary file {temp_path}: {str(e)}")

@app.route("/ats_check", methods=["POST"])
def ats_check():
    temp_path = None
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]
        if not file.filename.endswith((".pdf", ".docx")):
            return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

        # Create a temporary file with a unique name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_path = tmp.name
            file.save(temp_path)

        print(f"ATS check processing file: {file.filename} (saved as {temp_path})")

        # Check file format
        format_check = check_file_format(temp_path)

        # Extract text from the resume
        if temp_path.endswith(".pdf"):
            resume_text = extract_text_from_pdf(temp_path)
        elif temp_path.endswith(".docx"):
            resume_text = extract_text_from_docx(temp_path)
        else:
            resume_text = ""

        # Get optional job description and industry
        job_description = request.form.get("job_description", "")
        industry = request.form.get("industry", "")

        if not resume_text:
            return jsonify({"error": "Failed to extract text from resume"}), 400

        # Run ATS compatibility check
        ats_results = check_ats_compatibility(resume_text, job_description)

        # Add industry benchmarking if specified
        if industry:
            benchmark = benchmark_against_industry(resume_text, industry)
            if 'details' not in ats_results:
                ats_results['details'] = {}
            ats_results['details']['industry_benchmark'] = benchmark

            # Add industry-specific recommendations
            if 'missing_keywords' in benchmark and benchmark.get('industry_relevance', 0) < 0.5:
                missing_keywords = benchmark.get('missing_keywords', [])
                if missing_keywords and len(missing_keywords) > 0:
                    ats_results['recommendations'].append(
                        f"Add more {benchmark['industry']}-specific keywords: {', '.join(missing_keywords[:3])}"
                    )

        response = {
            "score": ats_results.get('score', 0),
            "warnings": ats_results.get('warnings', []),
            "recommendations": ats_results.get('recommendations', []),
            "file_format": {
                "format": format_check.get("format", "unknown"),
                "size_mb": format_check.get("size_mb", 0),
                "issues": format_check.get("issues", [])
            }
        }

        if 'details' in ats_results:
            response["details"] = ats_results['details']

        print("ATS Check Response:", response)
        return jsonify(response)
    except Exception as e:
        print(f"ATS check error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred during ATS check: {str(e)}"}), 500
    finally:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Temporary file {temp_path} removed successfully")
            except Exception as e:
                print(f"Error removing temporary file {temp_path}: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)