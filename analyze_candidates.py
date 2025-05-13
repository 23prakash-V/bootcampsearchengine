import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
import argparse
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Helper to extract year from citation
def extract_year(citation):
    match = re.search(r'(19|20)\d{2}', citation)
    return match.group(0) if match else None

# Helper to extract author/source from citation
def extract_author(citation):
    match = re.match(r'([A-Z][a-zA-Z\-]+)', citation)
    return match.group(0) if match else None

# Helper to count formatting in markup
def formatting_stats(markup):
    return {
        'bold': '<b>' in markup,
        'underline': '<u>' in markup,
        'highlight': '<hl>' in markup,
    }

def analyze_candidates(jsonl_file, predictions_file=None, score_threshold=0.5):
    # Load data
    candidates = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))
    df = pd.DataFrame(candidates)

    # If predictions file is present, load scores
    if predictions_file and os.path.exists(predictions_file):
        preds = []
        with open(predictions_file, 'r') as f:
            for line in f:
                preds.append(json.loads(line))
        if 'candidate' in preds[0]:
            scores = [p['score'] for p in preds]
        else:
            scores = [p['score'] for p in preds]
        df['score'] = scores

    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Number of candidates: {len(df)}")

    # Top 10 tags
    print("\n=== Top 10 Tags ===")
    tag_counts = df['tag'].value_counts().head(10)
    print(tag_counts)
    tag_counts.to_csv('analysis/top_tags.csv')

    # Top 10 citation authors/sources
    print("\n=== Top 10 Citation Authors/Sources ===")
    df['author'] = df['citation'].apply(extract_author)
    author_counts = df['author'].value_counts().head(10)
    print(author_counts)
    author_counts.to_csv('analysis/top_authors.csv')

    # Citations by year
    print("\n=== Citations by Year ===")
    df['year'] = df['citation'].apply(extract_year)
    year_counts = df['year'].value_counts().sort_index()
    print(year_counts)
    year_counts.to_csv('analysis/citations_by_year.csv')
    plt.figure(figsize=(10,4))
    year_counts.plot(kind='bar')
    plt.title('Citations by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('analysis/citations_by_year.png')
    plt.close()

    # Sentence and paragraph count per candidate
    print("\n=== Sentence and Paragraph Count ===")
    df['sentence_count'] = df['fulltext'].apply(lambda x: len(re.findall(r'\.', x)))
    df['paragraph_count'] = df['fulltext'].apply(lambda x: len(x.split('\n')))
    print(df[['sentence_count', 'paragraph_count']].describe())
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(df['sentence_count'], bins=30)
    plt.title('Sentence Count per Candidate')
    plt.subplot(1,2,2)
    sns.histplot(df['paragraph_count'], bins=30)
    plt.title('Paragraph Count per Candidate')
    plt.tight_layout()
    plt.savefig('analysis/sentence_paragraph_counts.png')
    plt.close()

    # Formatting analysis
    print("\n=== Formatting Analysis ===")
    fmt_stats = df['preserved_markup'].apply(formatting_stats)
    fmt_df = pd.DataFrame(list(fmt_stats))
    for fmt in ['bold', 'underline', 'highlight']:
        percent = 100 * fmt_df[fmt].mean()
        print(f"{fmt.title()}: {percent:.1f}% of candidates")
    fmt_df.to_csv('analysis/formatting_stats.csv')

    # Outlier detection for fulltext length
    print("\n=== Fulltext Length Outliers ===")
    print(df['fulltext'].str.len().describe())
    outliers = df[df['fulltext'].str.len() > df['fulltext'].str.len().quantile(0.99)]
    print(f"Top 1% longest candidates: {len(outliers)}")
    outliers[['tag','citation','fulltext']].to_csv('analysis/fulltext_outliers.csv')

    # Candidates per school/source (from file path if available)
    if 'source' in df.columns:
        print("\n=== Candidates per Source ===")
        print(df['source'].value_counts().head(10))
        df['source'].value_counts().to_csv('analysis/candidates_per_source.csv')

    # If predictions are present, analyze scores and simulate labels
    if 'score' in df.columns:
        print("\n=== Model Score Distribution ===")
        print(df['score'].describe())
        plt.figure(figsize=(8,4))
        sns.histplot(df['score'], bins=30)
        plt.title('Model Score Distribution')
        plt.xlabel('Score')
        plt.tight_layout()
        plt.savefig('analysis/model_score_distribution.png')
        plt.close()
        # Top/bottom N candidates
        print("\n=== Top 5 Candidates by Score ===")
        print(df.sort_values('score', ascending=False).head(5)[['tag','citation','score']])
        print("\n=== Bottom 5 Candidates by Score ===")
        print(df.sort_values('score', ascending=True).head(5)[['tag','citation','score']])
        df.sort_values('score', ascending=False).head(20).to_csv('analysis/top20_by_score.csv')
        df.sort_values('score', ascending=True).head(20).to_csv('analysis/bottom20_by_score.csv')

        # Simulated label encoding (not real ground truth!)
        df['label'] = (df['score'] >= score_threshold).astype(int)
        df['pred'] = df['label']  # Model predicts the same as label (since no ground truth)
        # Metrics
        precision = precision_score(df['label'], df['pred'])
        recall = recall_score(df['label'], df['pred'])
        accuracy = accuracy_score(df['label'], df['pred'])
        f1 = f1_score(df['label'], df['pred'])
        cm = confusion_matrix(df['label'], df['pred'])
        print("\n=== Simulated Metrics (using threshold, not real ground truth) ===")
        print(f"Threshold: {score_threshold}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1: {f1:.3f}")
        print(f"Confusion Matrix:\n{cm}")
        # Save confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative","Positive"])
        disp.plot()
        plt.title(f"Confusion Matrix (threshold={score_threshold})")
        plt.savefig('analysis/confusion_matrix.png')
        plt.close()
        # Save metrics
        with open('analysis/simulated_metrics.txt', 'w') as f:
            f.write(f"Threshold: {score_threshold}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"Accuracy: {accuracy:.3f}\n")
            f.write(f"F1: {f1:.3f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

    # Save updated DataFrame
    df.to_csv('analysis/candidate_full_analysis.csv', index=False)
    print("\nFull analysis saved to analysis/candidate_full_analysis.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_file', type=str, default='candidates.jsonl', help='Candidates JSONL file to analyze')
    parser.add_argument('--predictions_file', type=str, default=None, help='Predictions JSONL file (optional)')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Threshold for simulated label encoding')
    args = parser.parse_args()
    analyze_candidates(args.jsonl_file, args.predictions_file, args.score_threshold) 