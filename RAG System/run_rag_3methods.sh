#!/bin/bash

# 确保脚本在出错时停止执行
set -e

# 创建输出目录
mkdir -p system_outputs

# 设置文件路径变量
TRAIN_DIR="train"
TEST_DIR="test"
TXT_FILES_DIR="txt_files"
OUTPUT_DIR="system_outputs"
TRAIN_FILE="train_pairs.json" # 使用新的训练文件
QUESTIONS_FILE="questions.txt"

# 验证文件和目录是否存在
echo "正在验证必要的文件和目录..."

if [ ! -d "$TRAIN_DIR" ]; then
    echo "错误: 训练目录 $TRAIN_DIR 不存在!"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "错误: 测试目录 $TEST_DIR 不存在!"
    exit 1
fi

if [ ! -d "$TXT_FILES_DIR" ]; then
    echo "错误: 文本文件目录 $TXT_FILES_DIR 不存在!"
    exit 1
fi

if [ ! -f "$TRAIN_DIR/$TRAIN_FILE" ]; then
    echo "错误: 训练文件 $TRAIN_DIR/$TRAIN_FILE 不存在!"
    exit 1
fi

if [ ! -f "$TEST_DIR/$QUESTIONS_FILE" ]; then
    echo "错误: 问题文件 $TEST_DIR/$QUESTIONS_FILE 不存在!"
    exit 1
fi

# 检查questions.txt中是否有问题
QUESTION_COUNT=$(grep -c "[[:alnum:]]" "$TEST_DIR/$QUESTIONS_FILE" || true)
if [ "$QUESTION_COUNT" -eq 0 ]; then
    echo "错误: 问题文件 $TEST_DIR/$QUESTIONS_FILE 不包含任何问题!"
    exit 1
else
    echo "问题文件包含 $QUESTION_COUNT 个问题."
fi

# 打印一下测试文件前几行内容以供确认
echo "测试问题文件内容预览:"
head -n 3 "$TEST_DIR/$QUESTIONS_FILE"
echo "..."

# 运行RAG系统的三种不同配置

# 配置1: 默认配置加few-shot学习
echo "正在运行配置1: 默认配置加few-shot学习"
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

# 配置2: 更多文档, 不同模型
echo "正在运行配置2: 更多文档, 不同模型"
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

# 配置3: 优化配置
echo "正在运行配置3: 优化配置"
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

echo "所有RAG系统运行完成. 请检查 $OUTPUT_DIR 目录获取结果."

# 检查生成的输出文件
echo "验证输出文件..."
for i in {1..3}; do
    OUTPUT_FILE="$OUTPUT_DIR/system_output_$i.json"
    if [ -f "$OUTPUT_FILE" ]; then
        ANSWER_COUNT=$(python -c "import json; f=open('$OUTPUT_FILE'); data=json.load(f); print(len(data))")
        echo "系统输出 $i: 包含 $ANSWER_COUNT 个回答"
    else
        echo "警告: 系统输出 $i 文件不存在!"
    fi
done

echo "完成!"