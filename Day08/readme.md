## Resume Classification and Job Matching System

**Description:**  
A Machine Learning project that automatically classifies resumes into job categories and evaluates candidate-job fit. The system extracts and cleans text from resumes (PDF/DOCX), transforms it using TF-IDF, and predicts the relevant job domain using a trained SVM model. Additionally, it compares resumes with job descriptions using cosine similarity to provide a match percentage and highlight key missing or overlapping skills.

**Technologies Used:**  
- **Languages:** Python  
- **Libraries:** scikit-learn, pandas, numpy, PyPDF2, python-docx, re, pickle  
- **Machine Learning Techniques:** SVM classifier, TF-IDF vectorization, cosine similarity  

**Key Features:**  
- Automatic resume classification into predefined job domains.  
- Resume-to-job-description matching with similarity score.  
- Highlights missing or overlapping skills to improve candidate readiness.  

**Project Structure:**  
project/  
│── notebook.ipynb  
│── data/  
│ ├── clf.pkl  
│ ├── tfidf.pkl  
│ └── encoder.pkl  

**Impact:**  
- Helps recruiters quickly filter relevant resumes.  
- Assists job seekers in improving resumes for targeted roles.  
