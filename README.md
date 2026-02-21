# AI Resume Screening & Job Matching System

An NLP-powered web application that matches resumes with relevant job listings using transformer-based semantic similarity.

Built using Streamlit, Sentence-BERT, spaCy, and Scikit-learn.

# 1️⃣ Project Overview:

The AI Resume Screening & Job Matching System is a machine learning application that automatically ranks job listings based on how closely they match a candidate’s resume.

The system:

Processes unstructured resume text

Converts text into semantic embeddings using Sentence Transformers

Computes cosine similarity between resume and job descriptions

Ranks jobs based on relevance score

Displays top matching jobs in an interactive Streamlit web interface

This project demonstrates practical use of:

Natural Language Processing (NLP)

Transformer-based embeddings

Information retrieval techniques

Real-world recruitment automation

# 2️⃣ Problem Statement:

Recruiters manually screen hundreds of resumes for each job posting. This process is:

Time-consuming

Error-prone

Inconsistent

This project automates resume-job matching using semantic similarity scoring to reduce manual effort.

# 3️⃣ Solution Approach:

The system works in 5 main stages:

 1️⃣ Data Preprocessing:

Load job dataset (18K+ job listings)

Remove duplicates and short descriptions

Clean text using spaCy (lemmatization, stopword removal)

  2️⃣ Text Embedding:

Use all-MiniLM-L6-v2 Sentence-BERT model

Convert job descriptions into numerical vectors

Convert resume into embedding vector

  3️⃣ Similarity Computation:

Compute cosine similarity between:

Resume embedding
and
All job embeddings

  4️⃣ Ranking:

Multiply similarity score by 100

Sort jobs in descending order

Select top 5 matches

  5️⃣ User Interface (Streamlit):

Resume text input or PDF upload

Real-time matching

Display match percentage

Show top job recommendations

  4️⃣ Tech Stack:
    
Component	    Technology

Frontend	    Streamlit

NLP	spaCy,ML    Scikit-learn

Embeddings	    Sentence-Transformers

	            
Similarity	    Cosine Similarity

Language	    Python


   5️⃣ Project Structure:

Resume_Matcher/

├── app.py

├── job_data.pkl  

├── job_embeddings.npy   

├── requirements.txt  

└── README.md

   6️⃣ How to Run the Project

Run Streamlit App

streamlit run app.py


  7️⃣ Example Workflow

Input Resume:

Mechanical Engineer experienced in CAD, FEA, thermodynamics,

and manufacturing optimization.

Output:

Job Title	            Match             Score

Mechanical Engineer     FEA	              61%

Designer                II	              56%

Mechanical Engineer     Relocate	      54%

Higher score = stronger semantic similarity.

  8️⃣ Key Features

Resume text input

PDF resume upload

Semantic similarity ranking

Top 5 job recommendations

Match percentage visualization

Clean interactive UI

Real job dataset (18K+ entries)

  9️⃣ Core ML Concept Used

Cosine Similarity

Measures the angle between two vectors.

Value range: 0 to 1

1 → Identical meaning

0 → No similarity

Match Score = Cosine Similarity × 100

  🔟 Innovation Highlights

Uses transformer-based semantic embeddings

Handles unstructured resume text

Real-time ranking system

Deployable ML application

End-to-end pipeline from data preprocessing to UI

  1️⃣1️⃣ Future Improvements

Skill gap analysis

Skill highlighting

Recruiter dashboard

Multi-resume comparison

Deployment on cloud (AWS/Render)