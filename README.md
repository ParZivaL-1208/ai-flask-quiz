# AI Quiz Classifier Microservice (Flask)

This repository contains the Python-based AI microservice for the AI-Enhanced Quiz Platform. It handles PDF document ingestion, text extraction, and Natural Language Processing (NLP) to dynamically classify multiple-choice questions into specific chapters.



> **⚠️ CRITICAL ARCHITECTURE NOTE**
> 
> This is half of a decoupled system. It is designed to operate strictly as a backend processor for the main web application. 
> 
> **The full application requires two repositories running simultaneously:**
> 1. **This Python Microservice:** Handles machine learning and PDF processing (Runs on port 5000).
> 2. **The Spring Boot Main App:** Handles the UI, database, security, and user management (Runs on port 8080). You can find the main repository here: https://github.com/ParZivaL-1208/ai-quiz-platform.git
>
> If this Python server is offline, the Spring Boot application will throw `Connection refused` errors when attempting to upload materials or generate quizzes.

## ⚙️ How It Works

1. **Document Ingestion (`/upload_pdf`):** Receives PDF files from the Java backend, extracts raw text using `pdfplumber`, and cleans the data.
2. **Model Training:** Dynamically re-trains a `scikit-learn` TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer every time new material is uploaded.
3. **Classification (`/find_chapter`):** When the Java backend submits a new quiz question, this API calculates the cosine similarity between the question text and the ingested chapters, returning the most probable chapter origin. If the confidence score is too low, it flags the question as "Topic not found".

## 💻 Tech Stack

* **Framework:** Flask (Python)
* **Machine Learning:** `scikit-learn` (TF-IDF Vectorizer, Cosine Similarity)
* **Data Processing:** `numpy`
* **Document Parsing:** `pdfplumber`, `werkzeug`

## 🚀 Local Setup & Installation

You must run this service locally alongside the Spring Boot application.

### 1. Prerequisites
* Python 3.9+ installed on your machine.

### 2. Clone and Setup Environment
Navigate to the directory where you want to run the service:
```bash
git clone [https://github.com/ParZivaL-1208/ai-flask-quiz.git](https://github.com/ParZivaL-1208/ai-flask-quiz.git)
cd ai-flask-quiz

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
