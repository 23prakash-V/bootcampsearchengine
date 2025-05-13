Debate Evidence Extraction

This project builds a predictive pipeline to extract, format, and score debate evidence using different logistic regression & other models trained on high-quality, labeled evidence from external databases. The goal is to automate the identification of evidence blocks (tag, citation, fulltext, with formatting) for debate research and preparation (prep).

## Order of Events
1. Convert PDFs to DOCX externally via pdf2docx files if necessary
2. Extract Candidates: `extract_candidates_from_docx.py` parses DOCX files in `converted_docx/`, extracting candidate evidence blocks (tag, citation, fulltext, preserved markup) into `candidates.jsonl`.
3. Train Model: `train_evidence_model.py` trains classifier on labeled evidence data (from parquet files) and saves the model to `trained_model/`.
4. Predict Evidence: `predict_evidence.py` loads the trained model and scores each candidate in `candidates.jsonl`. 
5. Analyze Candidates: `analyze_candidates.py` provides EDA and visualizations of candidate extraction and prediction results. 
