#!/bin/bash
set -e
mkdir -p system_outputs
TRAIN_DIR="train"
TEST_DIR="test"
TXT_FILES_DIR="txt_files"
OUTPUT_DIR="system_outputs"
TRAIN_FILE="train_pairs.json"
QUESTIONS_FILE="questions_test.txt"  

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

# Check if the questions file contains any questions
QUESTION_COUNT=$(grep -c "[[:alnum:]]" "$TEST_DIR/$QUESTIONS_FILE" || true)
if [ "$QUESTION_COUNT" -eq 0 ]; then
    echo "Error: Questions file $TEST_DIR/$QUESTIONS_FILE does not contain any questions!"
    exit 1
else
    echo "Questions file contains $QUESTION_COUNT questions."
fi

# Print the first few lines of the test file for confirmation
echo "Test questions file preview:"
head -n 3 "$TEST_DIR/$QUESTIONS_FILE"
echo "..."

# Only run System 3 configuration (the best performing one)
echo "Running System 3 (optimized configuration with all-mpnet-base-v2 model)"
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

echo "System 3 run complete. Check the $OUTPUT_DIR directory for results."

# Check generated output file
echo "Validating output file..."
OUTPUT_FILE="$OUTPUT_DIR/system_output_1.json"
if [ -f "$OUTPUT_FILE" ]; then
    ANSWER_COUNT=$(python -c "import json; f=open('$OUTPUT_FILE'); data=json.load(f); print(len(data))")
    echo "System output: Contains $ANSWER_COUNT answers"
else
    echo "Warning: System output file does not exist!"
fi

echo "Complete!"