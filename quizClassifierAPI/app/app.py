import os
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
documents = {}
chapter_names = []
chapter_texts = []
text_vectorization_pipeline = None
chapter_vectors = None

def load_and_train_model():
    global documents, chapter_names, chapter_texts, text_vectorization_pipeline, chapter_vectors

    print("--- Starting model setup ---")

    # Clear previous data before reloading
    documents = {}
    chapter_names = []
    chapter_texts = []

    # Sort files alphabetically
    pdf_files = sorted([f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith('.pdf')])
    print(f"Found PDF files: {pdf_files}")

    if not pdf_files:
        print("WARNING: No PDF files found.")
        text_vectorization_pipeline = None
        chapter_vectors = None
        print("--- Model setup done. API is ready to requests. ---")
        return

    print("Step 1: Processing PDF files...")
    for filename in pdf_files:
        try:
            full_text = ""
            with pdfplumber.open(filename) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            documents[filename] = full_text
            print(f"  - Successfully processed: {filename}")
        except Exception as e:
            print(f"  - ERROR processing {filename}: {e}")

    if not documents:
        print("\nERROR: No PDF document successfully processed. The API cannot proceed.")
        text_vectorization_pipeline = None
        chapter_vectors = None
        return

    print(f"\nTotal documents loaded: {len(documents)}")

    chapter_names = list(documents.keys())
    chapter_texts = list(documents.values())

    print("Step 2: Training TF-IDF Vectorizer...")
    if chapter_texts:
        text_vectorization_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), max_df=0.85)),
        ])
        text_vectorization_pipeline.fit(chapter_texts)

        print("Step 3: Transforming chapter texts into vectors...")
        chapter_vectors = text_vectorization_pipeline.transform(chapter_texts)
    else:
        print("WARNING: No text content extracted from PDFs. Model cannot be trained.")
        text_vectorization_pipeline = None
        chapter_vectors = None

    print("\n--- Model setup complete. API is ready for requests. ---")

@app.route('/find_chapter', methods=['POST'])
def find_chapter():
    if text_vectorization_pipeline is None or chapter_vectors is None or not chapter_names:
        return jsonify({"error": "Model is not trained or no data available. Please upload PDF files."}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be a JSON"}), 400

    data = request.get_json()
    quiz_question = data.get('question')

    if not quiz_question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    quiz_question_str = str(quiz_question)
    question_vector = text_vectorization_pipeline.transform([quiz_question_str])

    similarities = cosine_similarity(question_vector, chapter_vectors)
    most_similar_chapter_index = np.argmax(similarities)
    best_match_filename = chapter_names[most_similar_chapter_index]

    # STRIP THE '.pdf' EXTENSION HERE
    if best_match_filename.lower().endswith('.pdf'):
        formatted_chapter = best_match_filename[:-4]
    else:
        formatted_chapter = best_match_filename

    highest_score = float(similarities[0, most_similar_chapter_index])

    threshold = 0.03
    if highest_score >= threshold:
        chapter_to_return = formatted_chapter
    else:
        chapter_to_return = f"Topic not found (but could be from: {formatted_chapter})"

    response = {
        "question": quiz_question_str,
        "most_likely_chapter": chapter_to_return,
        "similarity_score": round(highest_score, 4)
    }
    return jsonify(response)

@app.route('/upload_pdf', methods=['POST'])
def receive_pdf():
    save_path = None
    if 'fileUpload' not in request.files:
        print("!!! Upload Error: 'fileUpload' not in request.files !!!")
        return jsonify({"error": "No file part named 'fileUpload' in the request"}), 400

    file = request.files['fileUpload']

    if file.filename == '':
        print("!!! Upload Error: No file selected !!!")
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        try:
            filename = file.filename
            save_path = os.path.join('.', filename)
            file.save(save_path)
            print(f"--- Received and saved file: {filename} ---")

            print("--- Triggering model re-train... ---")
            load_and_train_model()

            if text_vectorization_pipeline is None or chapter_vectors is None:
                print("!!! ERROR: Model training failed after PDF upload. !!!")
                if os.path.exists(save_path):
                    os.remove(save_path)
                    print(f"--- Removed problematic file: {filename} ---")
                return jsonify({"error": "File uploaded, but model failed to re-train."}), 500

            print(f"--- Model re-trained successfully after uploading {filename} ---")
            return jsonify({"message": f"File '{filename}' uploaded and model updated successfully."}), 200
        except Exception as e:
            print(f"!!! ERROR processing uploaded file {file.filename}: {e} !!!")
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    print(f"--- Removed partially saved/problematic file: {save_path} ---")
                except Exception as remove_e:
                    print(f"!!! ERROR removing file {save_path}: {remove_e} !!!")
            return jsonify({"error": f"Could not process file: {e}"}), 500
    else:
        print(f"!!! Upload Error: Invalid file type '{file.filename}' !!!")
        return jsonify({"error": "Invalid file type, please upload a PDF."}), 400

if __name__ == '__main__':
    load_and_train_model()
    app.run(host='0.0.0.0', port=5000)