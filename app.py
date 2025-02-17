from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np
import docx
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load trained model and vectorizer
MODEL_PATH = r'C:\Users\nandl\Downloads\trained_model.pkl'
VECTORIZER_PATH = r'C:\Users\nandl\Downloads\vectorizer.pkl'

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Allowed file types
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from resume (TXT, DOCX, PDF)
def extract_text(file_path):
    text = ""
    
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = " ".join([p.text for p in doc.paragraphs])
    
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() if page.extract_text() else ''
    
    return text.strip()  # Ensure no empty spaces

# Function to calculate ATS score
def calculate_ats_score(resume_text):
    keywords = [
        "Skills", "Programming", "Languages", "Python", "Java", "JavaScript", "Machine learning",
        "Regression", "Cluster Analysis", "Neural Networks", "Database", "MySQL", "Tableau", "HTML",
        "CSS", "Angular", "Deep Learning", "Data Science", "Experience", "Company", "Technology",
        "Analytics", "Reports", "AI", "Models", "Algorithms", "NLP", "Big Data", "Visualization"
    ]
    
    resume_text = resume_text.lower()
    match_count = sum(1 for keyword in keywords if keyword.lower() in resume_text)
    
    ats_score = (match_count / len(keywords)) * 100 if keywords else 0  # Prevent division by zero
    return round(ats_score, 2)  # Keep it rounded to 2 decimal places

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Extract text from the resume
        resume_text = extract_text(file_path)
        
        if not resume_text:
            os.remove(file_path)  # Remove the file if no text was extracted
            return jsonify({'error': 'Could not extract text from the resume'}), 400

        # Transform text using vectorizer
        input_vector = vectorizer.transform([resume_text])
        
        # Make prediction
        prediction = model.predict(input_vector)
        
        # Ensure the prediction is a Python int (not numpy.int64)
        category_name = int(prediction[0]) if isinstance(prediction[0], np.integer) else str(prediction[0])

        # Calculate ATS score
        ats_score = calculate_ats_score(resume_text)
        
        os.remove(file_path)  # Clean up uploaded file
        
        return jsonify({'prediction': category_name, 'ats_score': float(ats_score)})
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)





  




    
    

