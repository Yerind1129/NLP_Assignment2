import json
import random
import os
import argparse
from typing import List, Dict, Any

def load_qa_pairs(input_file: str) -> List[Dict[str, Any]]:
    """Load QA pairs from the input file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_directory(directory: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_qa_pairs(qa_data: List[Dict[str, Any]], test_ratio: float = 0.1) -> tuple:
    """Split QA pairs into training and testing sets."""
    # Check if the input data is valid
    if not qa_data:
        print("Warning: Input data is empty!")
        return [], []

    # Validate the structure of each QA pair
    all_qa_pairs = []
    for item in qa_data:
        # Check if "source_file", "question", and "answer" keys exist
        if "source_file" not in item or "question" not in item or "answer" not in item:
            print(f"Warning: Skipping item due to missing keys: {item}")
            continue  # Skip this item

        all_qa_pairs.append({
            "source_file": item["source_file"],
            "question": item["question"],
            "answer": item["answer"]
        })
    
    print(f"Total valid QA pairs after filtering: {len(all_qa_pairs)}")
    
    # Shuffle the QA pairs
    random.shuffle(all_qa_pairs)
    
    # Split into train and test sets
    split_idx = int(len(all_qa_pairs) * (1 - test_ratio))
    train_pairs = all_qa_pairs[:split_idx]
    test_pairs = all_qa_pairs[split_idx:]
    
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    
    return train_pairs, test_pairs

def format_output_files(qa_pairs: List[Dict[str, Any]], output_dir: str):
    """Create formatted output files."""
    # Create questions.txt
    questions_file = os.path.join(output_dir, "questions.txt")
    with open(questions_file, 'w', encoding='utf-8') as f:
        for i, pair in enumerate(qa_pairs, 1):
            f.write(f"{pair['question']}\n")
    
    # Create reference_answers.json
    ref_answers = {}
    for i, pair in enumerate(qa_pairs, 1):
        ref_answers[str(i)] = pair["answer"]
    
    ref_file = os.path.join(output_dir, "reference_answers.json")
    with open(ref_file, 'w', encoding='utf-8') as f:
        json.dump(ref_answers, f, indent=2, ensure_ascii=False)
    
    # Save the full pairs as well (with source info) for reference
    full_pairs_file = os.path.join(output_dir, "full_qa_pairs.json")
    with open(full_pairs_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

def main(args):
    # Load QA pairs
    qa_data = load_qa_pairs(args.input_file)
    print(f"Loaded {len(qa_data)} items from input file.")
    
    # Create output directories
    create_directory(args.train_dir)
    create_directory(args.test_dir)
    
    # Split the data
    train_pairs, test_pairs = split_qa_pairs(qa_data, args.test_ratio)
    
    # Create formatted output files
    format_output_files(train_pairs, args.train_dir)
    format_output_files(test_pairs, args.test_dir)
    
    print(f"Split completed successfully!")
    print(f"Train set: {len(train_pairs)} QA pairs saved to {args.train_dir}")
    print(f"Test set: {len(test_pairs)} QA pairs saved to {args.test_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split QA pairs into train and test sets")
    parser.add_argument("--input_file", type=str, default="QA/qa_output_filtered/filtered_qa_pairs.json", 
                        help="Path to input JSON file with QA pairs")
    parser.add_argument("--train_dir", type=str, default="train", 
                        help="Directory to save training set files")
    parser.add_argument("--test_dir", type=str, default="test", 
                        help="Directory to save test set files")
    parser.add_argument("--test_ratio", type=float, default=0.1, 
                        help="Ratio of data to use for testing (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    main(args)