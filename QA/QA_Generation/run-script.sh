#!/bin/bash

# # Install required packages
# pip install torch transformers tqdm

# Run the script with optimized parameters for Llama 3 8B
# python qa_generation.py \
#   --input_dir="QA/txt_files" \
#   --output_dir="QA/qa_output" \
#   --model_name="/gpfsnyu/scratch/yx2432/models/llama-3.1-8b-instruct" \
#   --questions_per_file=9 \
#   --min_content_length=200 \
#   --batch_size=1

# Uncomment this to run validation after generation
python validation.py \
  --input_file="QA/qa_output/generated_qa_pairs.json" \
  --output_dir="QA/qa_output_filtered"
