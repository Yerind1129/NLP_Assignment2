#!/bin/bash
set -e
mkdir -p system_outputs
TRAIN_DIR="train"
TEST_DIR="test"
TXT_FILES_DIR="txt_files"
OUTPUT_DIR="system_outputs"
TRAIN_FILE="train_pairs.json"
QUESTIONS_FILE="questions.txt"
echo "Examining necessary directories and files"
if [ ! -d "$TRAIN_DIR" ]; then
    echo "Error: Training directory $TRAIN_DIR does not exist!"
    exit 1
fi
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory $TEST_DIR does not exist!"
    exit 1
fi
if [ ! -d "$TXT_FILES_DIR" ]; then
    echo "Error: Text files directory $TXT_FILES_DIR does not exist!"
    exit 1
fi
if [ ! -f "$TRAIN_DIR/$TRAIN_FILE" ]; then
    echo "Error: Training file $TRAIN_DIR/$TRAIN_FILE does not exist!"
    exit 1
fi
if [ ! -f "$TEST_DIR/$QUESTIONS_FILE" ]; then
    echo "Error: Questions file $TEST_DIR/$QUESTIONS_FILE does not exist!"
    exit 1
fi
# Check if questions.txt contains any questions
QUESTION_COUNT=$(grep -c "[[:alnum:]]" "$TEST_DIR/$QUESTIONS_FILE" || true)
if [ "$QUESTION_COUNT" -eq 0 ]; then
    echo "Error: Questions file $TEST_DIR/$QUESTIONS_FILE does not contain any questions!"
    exit 1
else
    echo "Questions file contains $QUESTION_COUNT questions."
fi
# Print the first few lines of the test file for confirmation
echo "Preview of test questions file content:"
head -n 3 "$TEST_DIR/$QUESTIONS_FILE"
echo "..."
# Run RAG system with three different configurations
# Configuration 1: Default configuration with few-shot learning
echo "Running configuration 1: Default configuration with few-shot learning"
python rag_system_fewshot.py \
    --train_dir "$TRAIN_DIR" \
    --train_file "$TRAIN_FILE" \
    --test_dir "$TEST_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --txt_files_dir "$TXT_FILES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --top_k 5 \
    --system_output_num 1 \
    --use_training_data
# Configuration 2: More documents, different model
echo "Running configuration 2: More documents, different model"
python rag_system_fewshot.py \
    --train_dir "$TRAIN_DIR" \
    --train_file "$TRAIN_FILE" \
    --test_dir "$TEST_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --txt_files_dir "$TXT_FILES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model sentence-transformers/multi-qa-mpnet-base-dot-v1 \
    --top_k 8 \
    --system_output_num 2 \
    --use_training_data
# Configuration 3: Optimized configuration
echo "Running configuration 3: Optimized configuration"
python rag_system_fewshot.py \
    --train_dir "$TRAIN_DIR" \
    --train_file "$TRAIN_FILE" \
    --test_dir "$TEST_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --txt_files_dir "$TXT_FILES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model sentence-transformers/all-mpnet-base-v2 \
    --top_k 3 \
    --system_output_num 3 \
    --use_training_data
echo "All RAG systems completed. Please check the $OUTPUT_DIR directory for results."
# Check generated output files
echo "Verifying output files..."
for i in {1..3}; do
    OUTPUT_FILE="$OUTPUT_DIR/system_output_$i.json"
    if [ -f "$OUTPUT_FILE" ]; then
        ANSWER_COUNT=$(python -c "import json; f=open('$OUTPUT_FILE'); data=json.load(f); print(len(data))")
        echo "System output $i: contains $ANSWER_COUNT answers"
    else
        echo "Warning: System output $i file does not exist!"
    fi
done
echo "Completed!"