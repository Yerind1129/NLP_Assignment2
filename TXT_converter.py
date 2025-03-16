import os
import json
import re

# dirs
INPUT_DIR = "/Users/dongyanxuan/Desktop/Spring2025/NLP/HW2/web_scraping/all_webs" # change this to the directory where you have all the folders with all_content.json files
OUTPUT_DIR = "/Users/dongyanxuan/Desktop/Spring2025/NLP/HW2/web_scraping/txt_files" # change this to the directory where you want to save the txt files

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_filename(filename):
    return re.sub(r'[^\w\-_.]', '_', filename)

def clean_text(text):

    cleaned = ''.join(char for char in text if char.isprintable())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def filter_username(text):
    return "username" not in text.lower()

def process_json_files():
    files_processed = 0
    paragraphs_filtered = 0
    
    for root, dirs, files in os.walk(INPUT_DIR):
        if OUTPUT_DIR in root:
            continue
            
        if 'all_content.json' in files:
            file_path = os.path.join(root, 'all_content.json')
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                folder_name = os.path.basename(root)
                
                for url, content_data in data.items():
                    if isinstance(content_data, dict) and 'title' in content_data and 'content' in content_data:
                        filtered_count = process_single_content(content_data, folder_name, url)
                        paragraphs_filtered += filtered_count
                        files_processed += 1
            
            except Exception as e:
                print(f"Error processing: {file_path}: {str(e)}")
    
    print(f"Done, converted {files_processed} txt files")
    print(f"Filtered {paragraphs_filtered} paragraphs containing 'username'")

def process_single_content(data, folder_name, url):

    title = data.get('title', 'Untitled')
    content_list = data.get('content', [])
    
    title = clean_text(title)
    
    if not content_list:
        return 0
    
    url_part = url.split('/')[-1] if '/' in url else url
    if not url_part:
        url_part = "index"
    
    txt_filename = f"{folder_name}_{clean_filename(url_part)}.txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    
    if len(txt_filename) > 100:
        txt_filename = f"{folder_name}_{clean_filename(url_part)[:50]}.txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    
    filtered_count = 0
    filtered_content = []
    for paragraph in content_list:
        clean_paragraph = clean_text(paragraph)
        if clean_paragraph and filter_username(clean_paragraph):
            filtered_content.append(clean_paragraph)
        elif clean_paragraph:
            filtered_count += 1
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n\n")
        
        for paragraph in filtered_content:
            f.write(f"{paragraph}\n\n")
    
    return filtered_count

if __name__ == "__main__":
    process_json_files()