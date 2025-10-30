# MSAI10_Projects — AI Course Assignments

**Author:** Salma Areef Syed  
**Course:** MSAI10 — Artificial Intelligence  
**Year:** 2025

This repository is the single source of truth for assignments and mini-projects completed for the AI course.  
Each module has its own folder under `modules/` and contains notebooks, scripts, data samples, reports, and results.

## Modules included
- Module 1 — Python Essentials (`modules/module1-python-essentials/`)
- Module 2 — Data Processing for AI (`modules/module2-data-processing/`)
- Module 3 — Statistical Foundations for AI (`modules/module3-statistical-foundations/`)
- Module 4 — Machine Learning (`modules/module4-machine-learning/`)
- Module 5 — Linear Algebra (`modules/module5-linear-algebra/`)

## How to reproduce / run
1. Clone the repository:
```bash
git clone https://github.com/SalAIStudy/MSAI10_Projects.git
cd MSAI10_Projects
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate.bat # Windows (PowerShell)
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Open any notebook:
```bash
jupyter notebook modules/module4-machine-learning/assignment.ipynb
```

## Submission checklist
- README.md (root) — this file
- modules/ — contains all module notebooks, scripts and data samples
- docs/design_document.md — design summary
- results/ — screenshots and metric summaries
- LICENSE, .gitignore, requirements.txt, submission_instructions.md

## Notes
- Large datasets and trained models are intentionally excluded. Where appropriate, small sample data is included under each module's `data/` folder and instructions are provided to reproduce results.
- If a notebook contains database or heavy-training code, the exported script has that code commented out. Original code cells are preserved for instructor review.
