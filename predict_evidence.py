import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import sys
import os

def debug(msg):
    print(f"[DEBUG] {msg}")

def load_candidates(candidates_file):
    debug(f"Loading candidates from {candidates_file}...")
    with open(candidates_file, 'r') as f:
        candidates = [json.loads(line) for line in f]
    debug(f"Loaded {len(candidates)} candidates.")
    return candidates

def predict(model_dir, candidates_file, output_file):
    debug(f"Loading model and tokenizer from {model_dir}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    debug(f"Using device: {device}")

    candidates = load_candidates(candidates_file)
    predictions = []
    for idx, candidate in enumerate(candidates):
        markup = candidate.get('preserved_markup') or candidate.get('fulltext') or candidate.get('markup')
        if not markup:
            debug(f"Candidate {idx} missing markup/fulltext field. Skipping.")
            continue
        inputs = tokenizer(markup, return_tensors="pt", truncation=True, padding='max_length', max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits.squeeze()).item()
        debug(f"Candidate {idx}: tag='{candidate.get('tag','')[:40]}', citation='{candidate.get('citation','')[:40]}', score={score:.4f}")
        predictions.append({"candidate": candidate, "score": score})
    debug(f"Saving predictions to {output_file}...")
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    debug(f"Done. {len(predictions)} predictions saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--candidates_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    predict(args.model_dir, args.candidates_file, args.output_file) 