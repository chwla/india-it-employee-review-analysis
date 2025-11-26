Deep Learning Based Resume to Job Description Matching

A Case Study on Indian Data Science Hiring

Overview

This project applies deep learning to improve the accuracy and speed of resume screening for Data Science roles in India. Indian companies receive a high volume of resumes, most of which vary widely in format, structure, and skill representation. Human recruiters spend significant time manually scanning these documents, which leads to inconsistent evaluations, overlooked candidates, and delays in the hiring process.

This project develops a simple but effective deep learning model that measures the match between a candidate’s resume and a job description. A baseline TF-IDF similarity model is included for comparison. The goal is not to build a commercial ATS, but to demonstrate how neural networks can enhance hiring decisions and deliver actionable insights for Indian businesses.

Business Problem

Most Indian companies, especially mid-sized firms and startups, do not have modern ATS systems. Recruiters review resumes manually, which leads to:

Inefficient screening time

Subjective evaluation

Misalignment between required and available skills

Poor matching between experience level and job requirements

A data-driven matching model can reduce these inefficiencies and serve as a decision support tool for hiring managers.

Objectives

Build a deep learning model that converts resumes and job descriptions into vector representations and predicts how well they match.

Create a simple TF-IDF and cosine similarity baseline for comparison.

Provide managerial insights about how such a system can support hiring in the Indian business environment.

Keep the solution lightweight and easy to deploy.

Dataset

The dataset includes real-world Indian Data Science job postings and sample resumes. Each record contains job role, company, location, experience range, and a skills or description field.
Since the dataset contains only Data Science roles, synthetic job descriptions were added to increase coverage across seniority levels. Resume-JD pairs were manually labeled as “good match” or “poor match” to train a supervised model.

Methodology

Data cleaning: lowercasing, removing noise, consolidating text fields.

Text preprocessing: tokenization and padding.

Deep learning model:

Embedding layer

LSTM encoder for resumes

LSTM encoder for job descriptions

Dense layers to predict match score

Baseline model: TF-IDF vectorization and cosine similarity.

Evaluation: accuracy, F1-score, confusion matrix, and qualitative examples.

Managerial analysis based on results and practical applicability.

Results

The deep learning model produced more stable match scores than the baseline, especially for resumes with inconsistent formatting or mixed skill descriptions. The LSTM-based model showed better sensitivity to contextual skill alignment and role expectations.

Managerial Insights

Screening time for recruiters can reduce significantly when a model provides an initial match score.

Hiring decisions become more consistent because every resume is evaluated against the same criteria.

Skill gaps can be identified across applicants, which helps with training and workforce planning.

The system can be integrated into internal HR workflows or used by placement teams in Indian universities.

Startups and mid-sized firms with limited HR bandwidth gain a practical tool that supports smarter hiring.

Folder Structure
project/
│── data/
│── notebooks/
│── models/
│── src/
│   ├── preprocessing.py
│   ├── train_lstm.py
│   ├── baseline_tfidf.py
│   ├── evaluate.py
│── results/
│── report/
│── README.md

How to Run

Install dependencies:
pip install -r requirements.txt

Place resumes and JDs in the data folder.

Run preprocessing:
python src/preprocessing.py

Train the LSTM model:
python src/train_lstm.py

Run baseline comparison:
python src/baseline_tfidf.py

View evaluation metrics in the results folder.

Conclusion

This project demonstrates that even a lightweight neural network can add measurable value to the hiring process in India. The combination of deep learning and simple NLP techniques supports faster, more consistent resume screening and offers insights that HR teams can use to improve recruitment efficiency.