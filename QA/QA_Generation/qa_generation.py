import os
import json
import argparse
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("qa_generation.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from text files using Llama 3")
    parser.add_argument("--input_dir", type=str, default="txt_files", help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", 
                       help="Hugging Face model name")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--questions_per_file", type=int, default=3, 
                       help="Number of questions to generate per file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing files")
    parser.add_argument("--min_content_length", type=int, default=100, 
                       help="Minimum character length to consider a file valid")
    return parser.parse_args()

def load_model(model_name):
    """Load model and tokenizer with proper settings for generation"""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure quantization for memory efficiency using bitsandbytes
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Set up model with appropriate parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # Use 4-bit quantization
        device_map="auto",  # Automatically distribute model across available GPUs
        trust_remote_code=True,
        max_memory={0: "24GiB"},  # Adjust based on your GPU memory
    )
    
    return model, tokenizer

def read_text_file(file_path):
    """Read a text file and return its content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        return content
    except Exception as e:
        logger.warning(f"Error reading file {file_path}: {e}")
        return ""

def is_valid_content(content, min_length=100):
    """Check if content is valid for processing"""
    # Simple validation: check if content is long enough
    return len(content) >= min_length

def generate_qa_pairs(model, tokenizer, content, num_questions=3):
    """Generate question-answer pairs from content"""
    # Truncate content if it's too long (Llama 3 8B has a context window of ~8K tokens)
    if len(content) > 6000:
        content = content[:6000] + "..."
    
    # Create a prompt that instructs the model to generate questions and answers about Pittsburgh and CMU
    system_prompt = "You are an expert on Pittsburgh and Carnegie Mellon University who creates high-quality test questions. Your task is to generate diverse questions and detailed answers about Pittsburgh and CMU."


    # For Llama 3, we'll use the chat format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Generate {num_questions} diverse high-quality questions and their detailed answers about Pittsburgh and Carnegie Mellon University.

    Your questions should cover the following categories, and please raise questions focus on the specific file you are processing:
    1. Simple factual questions that could be answered by just prompting an LLM
    2. Questions that can be better answered by augmenting an LLM with relevant documents
    3. Questions that are likely answered only through document augmentation
    4. Questions that are sensitive to temporal signals

    Include a mix and diverse of questions about:
    - Pittsburgh's history, geography, landmarks, and neighborhoods
    - Pittsburgh's culture, sports teams, and local events
    - Carnegie Mellon University's history, founding, and administration
    - CMU's academic programs, research centers, and notable achievements
    - Notable people associated with both Pittsburgh and CMU

    Format your response as follows:
    Q1: [Question 1]
    A1: [Answer 1]

    Q2: [Question 2]
    A2: [Answer 2]

    For questions have multiple answers, you can format as 
    Q3: [Question 3]
    A3: [Answer 3.1]; [Answer 3.2]; ...

    Make sure your answers are accurate and comprehensive.
    """}
    ]
    
    # Format messages according to Llama 3's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    response_text = generated_text.split("<assistant>")[-1].strip()
    
    # Parse Q&A pairs
    qa_pairs = []
    q_parts = response_text.split("Q")[1:]  # Split by question markers, skip first element
    
    for q_part in q_parts:
        try:
            # Extract question number and text
            q_num, rest = q_part.split(":", 1)
            q_num = q_num.strip()
            
            # Split the rest to get question and answer
            parts = rest.split("A" + q_num + ":", 1)
            
            if len(parts) != 2:
                # Try alternative format if the first attempt fails
                parts = rest.split("A" + q_num.strip() + ":", 1)
            
            if len(parts) != 2:
                # Try another format (just looking for A:)
                parts = rest.split("A:", 1)
            
            if len(parts) == 2:
                question = parts[0].strip()
                
                # Get the answer, handle cases where there might be another question
                answer_part = parts[1]
                if "Q" in answer_part and ":" in answer_part:
                    # Find the next Q marker
                    next_q_pos = answer_part.find("Q")
                    answer = answer_part[:next_q_pos].strip()
                else:
                    answer = answer_part.strip()
                
                qa_pairs.append({"question": question, "answer": answer})
        except Exception as e:
            logger.warning(f"Error parsing Q&A pair: {e}, Part: {q_part[:50]}...")
    
    return qa_pairs

def process_file(args, model, tokenizer, file_path):
    """Process a single file to generate Q&A pairs"""
    try:
        # Extract filename without extension for reference
        filename = os.path.basename(file_path)
        logger.info(f"Processing file: {filename}")
        
        # Read content
        content = read_text_file(file_path)
        
        # Skip if content is not valid
        if not is_valid_content(content, args.min_content_length):
            logger.info(f"Skipping {filename} - insufficient content")
            return None
        
        # Generate Q&A pairs
        qa_pairs = generate_qa_pairs(model, tokenizer, content, args.questions_per_file)
        
        # Skip if no pairs were generated
        if not qa_pairs:
            logger.info(f"No Q&A pairs generated for {filename}")
            return None
        
        # Clean up to save memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Return structured data
        result = {
            "source_file": filename,
            "qa_pairs": qa_pairs
        }
        
        # Log success
        logger.info(f"Successfully generated {len(qa_pairs)} Q&A pairs for {filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def save_output_files(all_qa_data, output_dir):
    """Save output files in various formats"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the complete QA data
    with open(os.path.join(output_dir, "generated_qa_pairs.json"), 'w', encoding='utf-8') as f:
        json.dump(all_qa_data, f, indent=2, ensure_ascii=False)
    
    # Save just the questions
    questions_file = os.path.join(output_dir, "questions.txt")
    with open(questions_file, 'w', encoding='utf-8') as f:
        for item in all_qa_data:
            for idx, qa_pair in enumerate(item["qa_pairs"]):
                f.write(f"{qa_pair['question']}\n")
    
    # Save reference answers
    reference_answers = {}
    for item in all_qa_data:
        for qa_pair in item["qa_pairs"]:
            reference_answers[qa_pair["question"]] = qa_pair["answer"]
    
    with open(os.path.join(output_dir, "reference_answers.json"), 'w', encoding='utf-8') as f:
        json.dump(reference_answers, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Output files saved to {output_dir}")
    logger.info(f"Total questions generated: {len(reference_answers)}")

def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get list of text files
    all_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                if f.endswith('.txt') and os.path.isfile(os.path.join(args.input_dir, f))]
    
    # Limit number of files if specified
    if args.max_files:
        all_files = all_files[:args.max_files]
    
    logger.info(f"Found {len(all_files)} text files for processing")
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name)
    
    # Process files
    all_qa_data = []
    
    # Save progress after each batch to prevent losing work if interrupted
    batch_size = min(10, len(all_files))  # Process and save in small batches
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        batch_results = []
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}")
        
        for file_path in tqdm(batch_files, desc=f"Batch {i//batch_size + 1}"):
            result = process_file(args, model, tokenizer, file_path)
            if result:
                batch_results.append(result)
                all_qa_data.append(result)
        
        # Save intermediate results
        if batch_results:
            interim_file = os.path.join(args.output_dir, f"qa_batch_{i//batch_size + 1}.json")
            with open(interim_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved batch results to {interim_file}")
            
        # Clear GPU memory between batches
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save final output files
    save_output_files(all_qa_data, args.output_dir)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()
