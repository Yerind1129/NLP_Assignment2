#!/usr/bin/env python3
"""
RAG Evaluation Script: Evaluates RAG system outputs against reference answers.
Computes standard metrics including exact match, F1, recall, and ROUGE scores.
"""

import json
import argparse
import os
import re
import numpy as np
from collections import Counter
from rouge import Rouge
import sys
from typing import Dict, List, Set, Tuple, Any, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def load_json(file_path: str) -> Dict[str, str]:
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)

def normalize_text(text: str) -> str:
    """Normalize text for fair comparison"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation for token-based metrics
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def get_tokens(text: str) -> List[str]:
    """Split text into tokens"""
    return normalize_text(text).split()

def calculate_exact_match(predictions: Dict[str, str], references: Dict[str, str]) -> float:
    """Calculate exact match between predictions and references"""
    keys = set(predictions.keys()).intersection(set(references.keys()))
    
    if not keys:
        return 0.0
    
    matches = 0
    for key in keys:
        if normalize_text(predictions[key]) == normalize_text(references[key]):
            matches += 1
    
    return matches / len(keys)

def calculate_token_f1(prediction_tokens: List[str], reference_tokens: List[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score at token level"""
    common_tokens = Counter(prediction_tokens) & Counter(reference_tokens)
    
    # Count the total tokens in the intersection
    num_common = sum(common_tokens.values())
    
    # If either is empty, return 0
    if len(prediction_tokens) == 0 or len(reference_tokens) == 0:
        return 0, 0, 0
    
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(reference_tokens)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

def calculate_f1_scores(predictions: Dict[str, str], references: Dict[str, str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 scores between predictions and references"""
    keys = set(predictions.keys()).intersection(set(references.keys()))
    
    if not keys:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for key in keys:
        pred_tokens = get_tokens(predictions[key])
        ref_tokens = get_tokens(references[key])
        
        precision, recall, f1 = calculate_token_f1(pred_tokens, ref_tokens)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    avg_precision = total_precision / len(keys)
    avg_recall = total_recall / len(keys)
    avg_f1 = total_f1 / len(keys)
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }

def calculate_rouge_scores(predictions: Dict[str, str], references: Dict[str, str]) -> Dict[str, float]:
    """Calculate ROUGE scores between predictions and references"""
    keys = set(predictions.keys()).intersection(set(references.keys()))
    
    if not keys:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    rouge = Rouge()
    
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for key in tqdm(keys, desc="Calculating ROUGE scores"):
        # Skip empty strings
        if not predictions[key] or not references[key]:
            continue
            
        try:
            scores = rouge.get_scores(predictions[key], references[key])[0]
            rouge_1_scores.append(scores['rouge-1']['f'])
            rouge_2_scores.append(scores['rouge-2']['f'])
            rouge_l_scores.append(scores['rouge-l']['f'])
        except Exception as e:
            print(f"Error calculating ROUGE for key {key}: {e}")
    
    if not rouge_1_scores:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    return {
        'rouge-1': np.mean(rouge_1_scores),
        'rouge-2': np.mean(rouge_2_scores),
        'rouge-l': np.mean(rouge_l_scores)
    }

def calculate_bleu_score(predictions: Dict[str, str], references: Dict[str, str]) -> float:
    """Calculate BLEU score between predictions and references"""
    keys = set(predictions.keys()).intersection(set(references.keys()))
    
    if not keys:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for key in tqdm(keys, desc="Calculating BLEU score"):
        reference = [references[key].split()]
        hypothesis = predictions[key].split()
        
        try:
            score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
            bleu_scores.append(score)
        except Exception as e:
            print(f"Error calculating BLEU for key {key}: {e}")
    
    if not bleu_scores:
        return 0.0
    
    return np.mean(bleu_scores)

def calculate_answer_coverage(predictions: Dict[str, str], references: Dict[str, str]) -> float:
    """Calculate the percentage of reference questions that have predictions"""
    if not references:
        return 0.0
    
    covered_keys = set(predictions.keys()).intersection(set(references.keys()))
    return len(covered_keys) / len(references)

def evaluate_all_outputs(output_dir: str, reference_file: str, output_file: Optional[str] = None) -> None:
    """Evaluate all system outputs found in the given directory"""
    # Load reference answers
    references = load_json(reference_file)
    print(f"Loaded {len(references)} reference answers from {reference_file}")
    
    # Find all system output files
    output_files = [f for f in os.listdir(output_dir) if f.startswith('system_output_') and f.endswith('.json')]
    
    if not output_files:
        print(f"No system output files found in {output_dir}")
        return
    
    print(f"Found {len(output_files)} system output files")
    
    # Evaluate each system output
    results = {}
    
    for output_file in sorted(output_files):
        output_path = os.path.join(output_dir, output_file)
        predictions = load_json(output_path)
        
        print(f"\nEvaluating {output_file} ({len(predictions)} predictions)...")
        
        # Calculate metrics
        exact_match = calculate_exact_match(predictions, references)
        f1_scores = calculate_f1_scores(predictions, references)
        rouge_scores = calculate_rouge_scores(predictions, references)
        bleu_score = calculate_bleu_score(predictions, references)
        answer_coverage = calculate_answer_coverage(predictions, references)
        
        # Compile results
        system_results = {
            'name': output_file,
            'predictions_count': len(predictions),
            'answer_coverage': answer_coverage,
            'exact_match': exact_match,
            'precision': f1_scores['precision'],
            'recall': f1_scores['recall'],
            'f1': f1_scores['f1'],
            'rouge-1': rouge_scores['rouge-1'],
            'rouge-2': rouge_scores['rouge-2'],
            'rouge-l': rouge_scores['rouge-l'],
            'bleu': bleu_score
        }
        
        results[output_file] = system_results
        
        # Print results for this system
        print("\nResults:")
        print(f"Answer Coverage: {answer_coverage:.4f}")
        print(f"Exact Match:     {exact_match:.4f}")
        print(f"Precision:       {f1_scores['precision']:.4f}")
        print(f"Recall:          {f1_scores['recall']:.4f}")
        print(f"F1 Score:        {f1_scores['f1']:.4f}")
        print(f"ROUGE-1:         {rouge_scores['rouge-1']:.4f}")
        print(f"ROUGE-2:         {rouge_scores['rouge-2']:.4f}")
        print(f"ROUGE-L:         {rouge_scores['rouge-l']:.4f}")
        print(f"BLEU:            {bleu_score:.4f}")
    
    # Determine the best system for each metric
    if len(results) > 1:
        metrics = ['exact_match', 'f1', 'rouge-1', 'rouge-l', 'bleu']
        best_systems = {}
        
        for metric in metrics:
            best_value = -1
            best_system = None
            
            for system, system_results in results.items():
                metric_value = system_results.get(metric, 0)
                if metric_value > best_value:
                    best_value = metric_value
                    best_system = system
            
            best_systems[metric] = (best_system, best_value)
        
        print("\n=== Best Systems by Metric ===")
        for metric, (system, value) in best_systems.items():
            print(f"{metric}: {system} ({value:.4f})")
    
    # Save results if output file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system outputs')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--predictions', type=str, help='Path to predictions JSON file')
    group.add_argument('--all', action='store_true', help='Evaluate all system outputs in system_outputs directory')
    
    parser.add_argument('--references', type=str, required=True, help='Path to reference answers JSON file')
    parser.add_argument('--output', type=str, help='Path to save evaluation results JSON file')
    parser.add_argument('--output_dir', type=str, default='system_outputs', help='Directory containing system outputs')
    
    args = parser.parse_args()
    
    try:
        # Check if NLTK and Rouge are installed
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt...")
            nltk.download('punkt', quiet=True)
    except ImportError:
        print("Error: NLTK is not installed. Please install it with: pip install nltk")
        sys.exit(1)
    
    try:
        import rouge
    except ImportError:
        print("Error: Rouge is not installed. Please install it with: pip install rouge")
        sys.exit(1)
    
    if args.all:
        # Evaluate all system outputs
        evaluate_all_outputs(args.output_dir, args.references, args.output)
    else:
        # Evaluate a single system output
        # Load predictions and references
        predictions = load_json(args.predictions)
        references = load_json(args.references)
        
        print(f"Loaded {len(predictions)} predictions and {len(references)} references")
        
        # Calculate metrics
        exact_match = calculate_exact_match(predictions, references)
        f1_scores = calculate_f1_scores(predictions, references)
        rouge_scores = calculate_rouge_scores(predictions, references)
        bleu_score = calculate_bleu_score(predictions, references)
        answer_coverage = calculate_answer_coverage(predictions, references)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Answer Coverage: {answer_coverage:.4f}")
        print(f"Exact Match:     {exact_match:.4f}")
        print(f"Precision:       {f1_scores['precision']:.4f}")
        print(f"Recall:          {f1_scores['recall']:.4f}")
        print(f"F1 Score:        {f1_scores['f1']:.4f}")
        print(f"ROUGE-1:         {rouge_scores['rouge-1']:.4f}")
        print(f"ROUGE-2:         {rouge_scores['rouge-2']:.4f}")
        print(f"ROUGE-L:         {rouge_scores['rouge-l']:.4f}")
        print(f"BLEU:            {bleu_score:.4f}")
        
        # Save results if output path is specified
        if args.output:
            results = {
                'exact_match': exact_match,
                'precision': f1_scores['precision'],
                'recall': f1_scores['recall'],
                'f1': f1_scores['f1'],
                'rouge-1': rouge_scores['rouge-1'],
                'rouge-2': rouge_scores['rouge-2'],
                'rouge-l': rouge_scores['rouge-l'],
                'bleu': bleu_score,
                'answer_coverage': answer_coverage
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
