# Design Document — MSAI10_Projects

## Repository Overview
This repository contains assignments and mini-projects completed as part of the Artificial Intelligence (AI) course.  
Each module folder in `modules/` corresponds to a major topic covered in the program.  
The repository follows the required structure outlined in the assignment submission guidelines, including README files, code, documentation, and results.

---

## 1. Modules Summary

### 🧩 Module 1 — Python Essentials
**Objective:**  
Learn Python programming basics required for AI, including variables, loops, functions, and simple data structures.

**Files:**  
- `modules/module1-python-essentials/assignment.ipynb`  
- `modules/module1-python-essentials/solution.py`  
- `modules/module1-python-essentials/data/`

**Key Highlights:**  
- Python syntax and control structures  
- File I/O, error handling, and basic algorithms  
- Simple examples demonstrating list comprehensions and functions  

---

### 🧮 Module 2 — Data Processing for AI
**Objective:**  
Understand and apply techniques for cleaning, transforming, and preparing data for AI and ML models.

**Files:**  
- `modules/module2-data-processing/assignment.ipynb`  
- `modules/module2-data-processing/preprocess.py`  
- `modules/module2-data-processing/data/`

**Key Highlights:**  
- Handling missing values and outliers  
- Data normalization and encoding  
- Exploratory Data Analysis (EDA) using `pandas` and `matplotlib`  

---

### 📊 Module 3 — Statistical Foundations for AI
**Objective:**  
Develop statistical intuition for AI, covering probability, hypothesis testing, and basic inference.

**Files:**  
- `modules/module3-statistical-foundations/assignment.ipynb`  
- `modules/module3-statistical-foundations/report.md`  

**Key Highlights:**  
- Probability distributions and expected values  
- Correlation, covariance, and sampling  
- Hypothesis testing and confidence intervals  

---

### 🤖 Module 4 — Machine Learning
**Objective:**  
Implement supervised and unsupervised learning models and evaluate their performance.

**Files:**  
- `modules/module4-machine-learning/assignment.ipynb`  
- `modules/module4-machine-learning/model.py`  
- `modules/module4-machine-learning/saved_models/`  
- `results/module4-machine-learning/`

**Key Highlights:**  
- Data splitting and model training  
- Algorithms: Linear Regression, Decision Tree, KNN (examples)  
- Evaluation metrics (accuracy, confusion matrix, RMSE)  

---

### 🔢 Module 5 — Linear Algebra
**Objective:**  
Understand linear algebra concepts used in AI models, such as matrices, vectors, and transformations.

**Files:**  
- `modules/module5-linear-algebra/assignment.ipynb`  
- `modules/module5-linear-algebra/exercises.py`

**Key Highlights:**  
- Vector and matrix operations using NumPy  
- Eigenvalues, eigenvectors, and matrix decompositions  
- Practical applications in ML (dimensionality reduction, PCA)  

---

## 2. Repository Execution Workflow
To reproduce or run any module assignment:

```bash
git clone https://github.com/SalAIStudy/MSAI10_Projects.git
cd MSAI10_Projects
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook
