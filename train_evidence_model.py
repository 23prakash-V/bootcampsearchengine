import os
import json
import re
import psutil
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def clean_text(text):
    """Clean and normalize text while preserving debate-specific patterns."""
    # Preserve citation patterns
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?\-()\[\]]', '', text)  # Keep parentheses and brackets
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Preserve formatting markers
    text = re.sub(r'\*\*(.*?)\*\*', r'[BOLD]\1[/BOLD]', text)
    text = re.sub(r'__(.*?)__', r'[UNDERLINE]\1[/UNDERLINE]', text)
    
    return text.strip()

def extract_debate_features(text):
    """Extract debate-specific features from text."""
    features = {}
    
    # Text structure features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
    
    # Citation features
    features['has_citation'] = 1 if re.search(r'\(\d{4}\)', text) else 0
    features['has_author'] = 1 if re.search(r'[A-Z][a-z]+,\s*[A-Z]', text) else 0
    features['citation_count'] = len(re.findall(r'\(\d{4}\)', text))
    
    # Evidence strength indicators
    strength_phrases = [
        'study shows', 'research indicates', 'according to', 'data suggests',
        'findings reveal', 'analysis shows', 'evidence suggests', 'demonstrates',
        'proves', 'confirms', 'indicates', 'reveals', 'shows that'
    ]
    features['strength_phrase_count'] = sum(1 for phrase in strength_phrases if phrase.lower() in text.lower())
    
    # Authority indicators
    authority_phrases = [
        'professor', 'director', 'institute', 'university', 'center for',
        'published in', 'journal of', 'research center', 'study by'
    ]
    features['authority_phrase_count'] = sum(1 for phrase in authority_phrases if phrase.lower() in text.lower())
    
    # Formatting features
    features['has_highlight'] = 1 if '_' in text or '*' in text else 0
    features['has_bold'] = 1 if '[BOLD]' in text else 0
    features['has_underline'] = 1 if '[UNDERLINE]' in text else 0
    
    # Number features
    features['num_count'] = len(re.findall(r'\d+', text))
    features['has_percentage'] = 1 if '%' in text else 0
    features['has_dollar'] = 1 if '$' in text else 0
    
    # Tag features
    features['has_section'] = 1 if re.search(r'\d+\.', text) else 0
    features['has_subsection'] = 1 if re.search(r'\.\d+$', text) else 0
    
    return features

def load_data_in_batches(jsonl_file, batch_size=1000):
    """Load data in batches to manage memory usage."""
    logger.info(f"Loading data from {jsonl_file}")
    
    # Count total lines
    total_lines = sum(1 for _ in open(jsonl_file))
    logger.info(f"Total lines: {total_lines:,}")
    
    # Process in batches
    train_data = []
    val_data = []
    processed_lines = 0
    error_count = 0
    
    with open(jsonl_file, 'r') as f:
        batch = []
        for line in tqdm(f, total=total_lines, desc="Loading data"):
            try:
                example = json.loads(line)
                text = example.get('text', '')
                if not isinstance(text, str):
                    error_count += 1
                    continue
                
                text = clean_text(text)
                label = example.get('label', 1)
                
                batch.append({
                    'text': text,
                    'label': int(label)
                })
                
                processed_lines += 1
                
                # Process batch when it reaches batch_size
                if len(batch) >= batch_size:
                    # Split batch into train/val
                    train_batch, val_batch = train_test_split(
                        batch, test_size=0.2, random_state=42,
                        stratify=[d['label'] for d in batch]
                    )
                    train_data.extend(train_batch)
                    val_data.extend(val_batch)
                    batch = []
                    
                    # Log progress
                    logger.info(f"Progress: {processed_lines:,}/{total_lines:,} lines "
                          f"({(processed_lines/total_lines*100):.1f}%)")
                    logger.info(f"Memory usage: {get_memory_usage():.2f} GB")
                    logger.info(f"Errors encountered: {error_count}")
            
            except json.JSONDecodeError as e:
                error_count += 1
                logger.error(f"JSON decode error: {str(e)}")
                continue
            except Exception as e:
                error_count += 1
                logger.error(f"Unexpected error: {str(e)}")
                continue
    
    # Process remaining items
    if batch:
        train_batch, val_batch = train_test_split(
            batch, test_size=0.2, random_state=42,
            stratify=[d['label'] for d in batch]
        )
        train_data.extend(train_batch)
        val_data.extend(val_batch)
    
    logger.info(f"Data loading complete. Total errors: {error_count}")
    return train_data, val_data

def train_model(train_data, val_data, output_dir):
    """Train model using debate-specific features and optimized parameters."""
    logger.info("Starting model training...")
    start_time = time.time()
    
    # Extract texts and labels
    train_texts = [d['text'] for d in train_data]
    train_labels = np.array([d['label'] for d in train_data])
    val_texts = [d['text'] for d in val_data]
    val_labels = np.array([d['label'] for d in val_data])
    
    # Create debate-specific TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 3),
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm='l2',
        token_pattern=r'(?u)\b\w\w+\b|(?:\d{4})|(?:\[BOLD\].*?\[/BOLD\])|(?:\[UNDERLINE\].*?\[/UNDERLINE\])'
    )
    
    # Fit and transform training data
    logger.info("Vectorizing training data...")
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_val_tfidf = vectorizer.transform(val_texts)
    
    # Extract debate-specific features
    logger.info("Extracting debate-specific features...")
    X_train_custom = np.array([list(extract_debate_features(text).values()) for text in train_texts])
    X_val_custom = np.array([list(extract_debate_features(text).values()) for text in val_texts])
    
    # Combine TF-IDF and custom features using sparse matrices
    X_train = sparse.hstack([X_train_tfidf, sparse.csr_matrix(X_train_custom)])
    X_val = sparse.hstack([X_val_tfidf, sparse.csr_matrix(X_val_custom)])
    
    # Train Random Forest model with optimized parameters
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        bootstrap=True,
        class_weight='balanced',
        verbose=1,
        warm_start=True  # Enable warm start for potential early stopping
    )
    
    # Perform cross-validation
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, train_labels, cv=5, scoring='f1')
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    model.fit(X_train, train_labels)
    
    # Evaluate on validation set
    logger.info("Evaluating model...")
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    precision = precision_score(val_labels, y_pred)
    recall = recall_score(val_labels, y_pred)
    accuracy = accuracy_score(val_labels, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    conf_matrix = confusion_matrix(val_labels, y_pred)
    
    # Log detailed evaluation results
    logger.info("\nEvaluation Results:")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info("\nConfusion Matrix:")
    logger.info(conf_matrix)
    logger.info("\nClassification Report:")
    logger.info(classification_report(val_labels, y_pred))
    
    # Feature importance analysis
    feature_importance = model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
    logger.info("\nTop 10 most important features:")
    for idx in top_features_idx:
        logger.info(f"Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Save model and vectorizer
    logger.info(f"Saving model and vectorizer to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f'model_{timestamp}.joblib')
    vectorizer_path = os.path.join(output_dir, f'vectorizer_{timestamp}.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save evaluation results
    eval_results = {
        'timestamp': timestamp,
        'training_time': time.time() - start_time,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': feature_importance.tolist(),
        'top_features': top_features_idx.tolist()
    }
    
    eval_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Vectorizer saved to: {vectorizer_path}")
    logger.info(f"Evaluation results saved to: {eval_path}")

if __name__ == "__main__":
    input_file = "parquet_with_negatives.jsonl"
    output_dir = "trained_model"
    
    logger.info("Starting training process...")
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    try:
        train_data, val_data = load_data_in_batches(input_file)
        train_model(train_data, val_data, output_dir) 
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise 
    finally:
        logger.info("Training process completed.")