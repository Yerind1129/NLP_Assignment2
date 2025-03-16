import json
import os
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate and filter generated Q&A pairs")
    parser.add_argument("--input_file", type=str, default="qa_output/generated_qa_pairs.json",
                        help="Input JSON file with generated Q&A pairs")
    parser.add_argument("--output_dir", type=str, default="qa_output_filtered",
                        help="Directory to save filtered output files")
    parser.add_argument("--min_question_length", type=int, default=20,
                        help="Minimum length of valid questions")
    parser.add_argument("--min_answer_length", type=int, default=50,
                        help="Minimum length of valid answers")
    parser.add_argument("--max_duplicate_similarity", type=float, default=0.85,
                        help="Maximum similarity threshold for detecting duplicates")
    return parser.parse_args()

def is_valid_qa_pair(qa_pair, min_q_len=20, min_a_len=50):
    """Check if a QA pair meets quality criteria"""
    question = qa_pair["question"]
    answer = qa_pair["answer"]
    
    # Length checks
    # if len(question) < min_q_len:
    #     return False, "Question too short"
    
    # if len(answer) < min_a_len:
    #     return False, "Answer too short"
    
    # Check for question marks in question
    if "?" not in question:
        return False, "No question mark in question"
    
    # Check for generic/template questions that weren't properly filled
    template_indicators = ["[", "]", "{", "}", "placeholder"]
    if any(indicator in question.lower() for indicator in template_indicators):
        return False, "Contains template markers"
    
    return True, "Valid"

def simple_similarity(str1, str2):
    """Calculate a simple similarity score between two strings"""
    # Convert to lowercase and split into words
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def detect_duplicates(qa_pairs, max_similarity=0.85):
    """Detect and mark duplicate questions"""
    unique_pairs = []
    duplicates = []
    
    for i, qa_pair in enumerate(qa_pairs):
        is_duplicate = False
        current_q = qa_pair["question"]
        
        for unique_pair in unique_pairs:
            existing_q = unique_pair["question"]
            similarity = simple_similarity(current_q, existing_q)
            
            if similarity > max_similarity:
                is_duplicate = True
                duplicates.append({
                    "index": i,
                    "question": current_q,
                    "similar_to": existing_q,
                    "similarity": similarity
                })
                break
        
        if not is_duplicate:
            unique_pairs.append(qa_pair)
    
    return unique_pairs, duplicates

def group_by_source(qa_data):
    """Group QA pairs by source file"""
    source_groups = defaultdict(list)
    
    for item in qa_data:
        source_file = item["source_file"]
        for qa_pair in item["qa_pairs"]:
            source_groups[source_file].append(qa_pair)
    
    return source_groups

def main():
    args = parse_arguments()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load generated QA pairs
    with open(args.input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # Flatten the QA pairs for processing
    all_qa_pairs = []
    for item in qa_data:
        for qa_pair in item["qa_pairs"]:
            all_qa_pairs.append({
                "source_file": item["source_file"],
                "question": qa_pair["question"],
                "answer": qa_pair["answer"]
            })
    
    print(f"Total QA pairs: {len(all_qa_pairs)}")
    
    # Validate QA pairs
    valid_qa_pairs = []
    invalid_qa_pairs = []
    
    for pair in all_qa_pairs:
        is_valid, reason = is_valid_qa_pair(
            pair, args.min_question_length, args.min_answer_length
        )
        
        if is_valid:
            valid_qa_pairs.append(pair)
        else:
            invalid_qa_pairs.append({"pair": pair, "reason": reason})
    
    print(f"Valid QA pairs: {len(valid_qa_pairs)}")
    print(f"Invalid QA pairs: {len(invalid_qa_pairs)}")
    
    # Detect duplicates
    unique_qa_pairs, duplicates = detect_duplicates(valid_qa_pairs, args.max_duplicate_similarity)
    print(f"Unique QA pairs after duplicate removal: {len(unique_qa_pairs)}")
    print(f"Duplicates found: {len(duplicates)}")
    
    # Save filtered output files
    
    # Save all unique QA pairs
    with open(os.path.join(args.output_dir, "filtered_qa_pairs.json"), 'w', encoding='utf-8') as f:
        json.dump(unique_qa_pairs, f, indent=2, ensure_ascii=False)
    
    # Save questions.txt
    with open(os.path.join(args.output_dir, "questions.txt"), 'w', encoding='utf-8') as f:
        for qa_pair in unique_qa_pairs:
            f.write(f"{qa_pair['question']}\n")
    
    # Save reference_answers.json
    reference_answers = {}
    for qa_pair in unique_qa_pairs:
        reference_answers[qa_pair["question"]] = qa_pair["answer"]
    
    with open(os.path.join(args.output_dir, "reference_answers.json"), 'w', encoding='utf-8') as f:
        json.dump(reference_answers, f, indent=2, ensure_ascii=False)
    
    # Save validation report
    validation_report = {
        "total_pairs": len(all_qa_pairs),
        "valid_pairs": len(valid_qa_pairs),
        "invalid_pairs": invalid_qa_pairs,
        "duplicates": duplicates,
        "final_count": len(unique_qa_pairs)
    }
    
    with open(os.path.join(args.output_dir, "validation_report.json"), 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    
    print(f"Filtered QA pairs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
