import os
import json
import random
import re
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'negative_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_negative_examples(markup: str, max_negatives: int = 1) -> List[Dict[str, Any]]:
    """Generate negative examples by manipulating the markup structure."""
    negatives = []
    parts = markup.split('\n')
    logger.info(f"Processing markup with {len(parts)} parts")
    
    if len(parts) < 3:
        logger.info("Skipping - not enough parts")
        return negatives
        
    tag = parts[0].strip()
    cite = parts[1].strip()
    fulltext = '\n'.join(parts[2:]).strip()
    
    logger.info(f"Tag: {tag[:50]}...")
    logger.info(f"Cite: {cite[:50]}...")
    logger.info(f"Fulltext length: {len(fulltext)}")
    
    # Define all possible negative transformations
    transformations = []
    
    # 1. Reorder components (tag, cite, fulltext)
    if len(parts) >= 3:
        # Move cite to end
        reordered1 = f"{tag}\n{fulltext}\n{cite}"
        transformations.append({
            'markup': reordered1,
            'reason': 'reordered_components'
        })
        
        # Move tag to end
        reordered2 = f"{cite}\n{fulltext}\n{tag}"
        transformations.append({
            'markup': reordered2,
            'reason': 'reordered_components'
        })
        logger.info("Added reordering transformations")
    
    # 2. Mix evidence blocks (for longer texts)
    if len(fulltext) > 100:
        sentences = fulltext.split('. ')
        if len(sentences) > 2:
            # Take first half of one evidence and second half of another
            mid = len(sentences) // 2
            mixed_text = '. '.join(sentences[mid:] + sentences[:mid])
            mixed_markup = f"{tag}\n{cite}\n{mixed_text}"
            transformations.append({
                'markup': mixed_markup,
                'reason': 'mixed_evidence'
            })
            logger.info("Added mixing transformation")
    
    # 3. Incomplete evidence
    if len(parts) >= 3:
        # Missing tag
        no_tag = f"{cite}\n{fulltext}"
        transformations.append({
            'markup': no_tag,
            'reason': 'missing_tag'
        })
        
        # Missing cite
        no_cite = f"{tag}\n{fulltext}"
        transformations.append({
            'markup': no_cite,
            'reason': 'missing_cite'
        })
        logger.info("Added incomplete evidence transformations")
    
    # 4. Corrupted formatting
    if len(parts) >= 3:
        # Add random newlines in fulltext
        corrupted_text = fulltext.replace('. ', '.\n\n')
        corrupted_markup = f"{tag}\n{cite}\n{corrupted_text}"
        transformations.append({
            'markup': corrupted_markup,
            'reason': 'corrupted_formatting'
        })
        logger.info("Added corrupted formatting transformation")
    
    # Always generate at least one negative example if we have transformations
    if transformations:
        logger.info(f"Found {len(transformations)} possible transformations")
        # If we have fewer transformations than max_negatives, use all of them
        # Otherwise randomly select max_negatives
        selected = random.sample(transformations, min(max_negatives, len(transformations)))
        for trans in selected:
            trans['label'] = 0
            negatives.append(trans)
        logger.info(f"Generated {len(negatives)} negative examples")
    else:
        logger.info("No transformations were possible")
    
    return negatives

def process_parquet_files(input_dir: str, output_file: str):
    """Process parquet files and generate negative examples."""
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    logger.info(f"Found {len(parquet_files)} parquet files in {input_dir}")
    
    all_examples = []
    positive_count = 0
    negative_count = 0
    
    for file_name in parquet_files:
        file_path = os.path.join(input_dir, file_name)
        logger.info(f"Processing {file_path}...")
        df = pd.read_parquet(file_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{file_name}"):
            # Get all fields from the parquet file
            tag = (row.get('tag') or '').strip()
            cite = (row.get('cite') or '').strip()
            fulltext = (row.get('fulltext') or '').strip()
            
            # Skip if we don't have all required fields
            if not (tag and cite and fulltext):
                continue
            
            # Construct markup from individual fields
            markup = f"{tag}\n{cite}\n{fulltext}"
            
            # Add positive example with markup as primary signal
            example = {
                'markup': markup,
                'tag': tag,
                'cite': cite,
                'fulltext': fulltext,
                'label': 1
            }
            
            all_examples.append(example)
            positive_count += 1
            
            # Generate negative examples by manipulating markup
            negatives = generate_negative_examples(markup, max_negatives=2)  # Generate 2 negatives per positive
            for neg in negatives:
                neg['tag'] = tag  # Keep original fields for reference
                neg['cite'] = cite
                neg['fulltext'] = fulltext
            all_examples.extend(negatives)
            negative_count += len(negatives)
            
            # Log progress periodically
            if positive_count % 1000 == 0:
                logger.info(f"Processed {positive_count} positive examples, generated {negative_count} negative examples")
                logger.info(f"Current ratio: {negative_count/positive_count:.2f} negatives per positive")
    
    # Shuffle all examples
    random.shuffle(all_examples)
    
    # Write to file
    logger.info(f"Writing {len(all_examples)} examples to {output_file}")
    with open(output_file, 'w') as f:
        for example in tqdm(all_examples, desc="Writing examples"):
            f.write(json.dumps(example) + '\n')
    
    logger.info("\nFinal Statistics:")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Positive examples: {positive_count}")
    logger.info(f"Negative examples: {negative_count}")
    logger.info(f"Ratio: {negative_count/positive_count:.2f} negatives per positive")

if __name__ == "__main__":
    input_dir = "parquet_data"
    output_file = "parquet_with_negatives.jsonl"
    process_parquet_files(input_dir, output_file) 