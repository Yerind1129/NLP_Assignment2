import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import argparse
import faiss
import sys

class RAGSystem:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, train_dir=None, train_file=None):
        """
        Initialize the RAG system with the specified embedding model.
        
        Args:
            model_name: The name of the sentence transformer model to use
            device: The device to run the model on (cuda or cpu)
            train_dir: Directory containing training data
            train_file: Name of the training file in train directory
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize the embedding model
        self.embedder = SentenceTransformer(model_name, device=self.device)
        
        # Initialize the document store
        self.documents = []
        self.document_embeddings = None
        self.index = None
        
        # Store training examples for few-shot learning
        self.train_examples = []
        self.train_question_embeddings = None
        self.train_dir = train_dir
        self.train_file = train_file or "full_qa_pairs.json"
        
        # Load training data if provided
        if train_dir:
            self.load_training_data(train_dir, self.train_file)
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize LLM for generating answers
        # We'll use a local LLM via Ollama for the reader component
        # You can replace this with any other LLM API
        try:
            # Try to import the updated version first
            try:
                from langchain_community.llms import Ollama
                self.llm = Ollama(model="llama2")
                self.has_llm = True
                print("Initialized local LLM via Ollama (community)")
            except ImportError:
                # Fall back to the deprecated version if needed
                from langchain.llms import Ollama
                self.llm = Ollama(model="llama2")
                self.has_llm = True
                print("Initialized local LLM via Ollama (legacy)")
        except Exception as e:
            print(f"Warning: Could not initialize Ollama: {e}")
            print("Falling back to retrieval-based answers.")
            self.has_llm = False
    
    def load_training_data(self, train_dir, train_file="full_qa_pairs.json"):
        """Load training data for few-shot learning"""
        qa_file = os.path.join(train_dir, train_file)
        
        if not os.path.exists(qa_file):
            print(f"Warning: Training data file {qa_file} not found")
            return
        
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
                
            # Clean up training data
            for qa_pair in qa_pairs:
                if isinstance(qa_pair, dict) and "question" in qa_pair and "answer" in qa_pair:
                    # Remove special markers
                    question = qa_pair["question"].replace("**", "").strip()
                    answer = qa_pair["answer"].replace("**", "").strip()
                    
                    self.train_examples.append({
                        "question": question,
                        "answer": answer
                    })
            
            print(f"Loaded {len(self.train_examples)} training examples from {qa_file}")
            
            # Pre-compute embeddings for training questions
            if self.train_examples:
                train_questions = [ex["question"] for ex in self.train_examples]
                self.train_question_embeddings = self.embed_texts(train_questions)
        
        except Exception as e:
            print(f"Error loading training data: {e}")
    
    def add_documents(self, documents, document_ids=None):
        """
        Add documents to the document store and create embeddings.
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs
        """
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
            
        # Split documents into chunks
        all_chunks = []
        all_chunk_ids = []
        
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            chunk_ids = [f"{document_ids[i]}_chunk_{j}" for j in range(len(chunks))]
            
            all_chunks.extend(chunks)
            all_chunk_ids.extend(chunk_ids)
        
        # Store documents with their IDs
        self.documents = list(zip(all_chunk_ids, all_chunks))
        
        # Create embeddings for documents
        print(f"Creating embeddings for {len(all_chunks)} document chunks...")
        self.document_embeddings = self.embed_texts(all_chunks)
        
        # Initialize FAISS index for fast similarity search
        self._create_index()
        
        print(f"Added {len(documents)} documents ({len(all_chunks)} chunks) to the document store")
        
    def _create_index(self):
        """Create a FAISS index for fast similarity search"""
        vector_dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(self.document_embeddings)
    
    def embed_texts(self, texts):
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        return self.embedder.encode(texts, convert_to_numpy=True)
    
    def get_similar_training_examples(self, query, k=3):
        """Get the k most similar training examples to the given query"""
        if not self.train_examples or self.train_question_embeddings is None:
            return []
        
        # Embed the query
        query_embedding = self.embed_texts([query])[0]
        
        # Calculate similarities with training questions
        similarities = []
        for i, embedding in enumerate(self.train_question_embeddings):
            # Compute cosine similarity
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((i, sim))
        
        # Get top-k similar examples
        top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        
        return [self.train_examples[idx] for idx, _ in top_indices]
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document_id, document_text, similarity_score) tuples
        """
        # Get similar training examples to enhance retrieval
        similar_examples = self.get_similar_training_examples(query, k=1)
        
        # If we found a very similar training example, use its question to enhance retrieval
        enhanced_query = query
        if similar_examples and len(similar_examples) > 0:
            example = similar_examples[0]
            if example["question"] != query:  # Don't use the exact same question
                enhanced_query = f"{query} {example['question']}"
        
        # Embed the enhanced query
        query_embedding = self.embed_texts([enhanced_query])
        
        # Search for similar documents
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the retrieved documents
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure index is valid
                doc_id, doc_text = self.documents[idx]
                retrieved_docs.append((doc_id, doc_text, 1.0 - distances[0][i]/2))  # Convert L2 distance to similarity score
        
        return retrieved_docs
    
    def generate_answer(self, query, retrieved_docs, max_length=512):
        """
        Generate an answer for the query based on retrieved documents.
        
        Args:
            query: Query text
            retrieved_docs: List of retrieved documents
            max_length: Maximum length of the generated answer
            
        Returns:
            Generated answer string
        """
        if not self.has_llm:
            # Fallback to simple retrieval-based answer
            if retrieved_docs:
                return f"Based on retrieved information: {retrieved_docs[0][1][:400]}..."
            return "No relevant information found."
        
        # Get few-shot examples
        examples = self.get_similar_training_examples(query, k=2)
        
        # Create examples text for the prompt
        examples_text = ""
        for i, example in enumerate(examples):
            examples_text += f"Example {i+1}:\nQuestion: {example['question']}\nAnswer: {example['answer']}\n\n"
        
        # Create context from retrieved documents
        context = "\n\n".join([f"Document: {doc[1]}" for doc in retrieved_docs])
        
        # Create prompt for the LLM with few-shot examples
        prompt = f"""You are a helpful assistant that answers questions about Pittsburgh and CMU (Carnegie Mellon University).
Answer the following question based ONLY on the provided context and examples. 
Be concise and to the point. Provide factual information from the context.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Here are some examples of good question-answer pairs:

{examples_text}
Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        try:
            answer = self.llm.invoke(prompt, max_tokens=max_length)
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            # Fallback: check if we have a similar training example
            if examples:
                # Return the answer from the most similar example
                return f"Based on similar questions: {examples[0]['answer']}"
            # Otherwise return document text
            if retrieved_docs:
                return f"Based on retrieved information: {retrieved_docs[0][1][:400]}..."
            return "No relevant information found."
    
    def answer_question(self, query, top_k=5):
        """
        End-to-end pipeline to answer a question.
        
        Args:
            query: Question text
            top_k: Number of documents to retrieve
            
        Returns:
            Generated answer string and retrieved documents
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        return answer, retrieved_docs

def load_json(file_path):
    """Load data from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_questions(file_path):
    """Load questions from a text file"""
    if not os.path.exists(file_path):
        print(f"ERROR: Questions file {file_path} does not exist! Please check the file path.")
        return []
        
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(questions)} questions from {file_path}")
    return questions

def main():
    parser = argparse.ArgumentParser(description='Run RAG system on local files')
    parser.add_argument('--train_dir', type=str, default='train', help='Directory containing training data')
    parser.add_argument('--train_file', type=str, default='full_qa_pairs.json', help='Name of the training file in train directory')
    parser.add_argument('--test_dir', type=str, default='test', help='Directory containing test data')
    parser.add_argument('--questions_file', type=str, default='questions.txt', help='Name of the questions file in test directory')
    parser.add_argument('--txt_files_dir', type=str, default='txt_files', help='Directory containing scraped text files')
    parser.add_argument('--output_dir', type=str, default='system_outputs', help='Directory to save outputs')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Sentence transformer model to use')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cuda or cpu)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of documents to retrieve for each question')
    parser.add_argument('--system_output_num', type=int, default=1, help='System output number (1, 2, or 3)')
    parser.add_argument('--use_training_data', action='store_true', help='Use training data for few-shot learning')
    
    args = parser.parse_args()
    
    # Validate directories and files
    for dir_path in [args.train_dir, args.test_dir, args.txt_files_dir]:
        if not os.path.isdir(dir_path):
            print(f"ERROR: Directory {dir_path} does not exist!")
            sys.exit(1)
    
    # Check required files
    if args.use_training_data:
        train_file_path = os.path.join(args.train_dir, args.train_file)
        if not os.path.exists(train_file_path):
            print(f"ERROR: Training file {train_file_path} does not exist!")
            sys.exit(1)
    
    questions_file_path = os.path.join(args.test_dir, args.questions_file)
    if not os.path.exists(questions_file_path):
        print(f"ERROR: Questions file {questions_file_path} does not exist!")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the RAG system
    train_dir = args.train_dir if args.use_training_data else None
    train_file = args.train_file if args.use_training_data else None
    
    rag_system = RAGSystem(
        model_name=args.model, 
        device=args.device, 
        train_dir=train_dir,
        train_file=train_file
    )
    
    # Load documents from txt_files directory
    print(f"Loading documents from {args.txt_files_dir}...")
    document_files = [os.path.join(args.txt_files_dir, fname) for fname in os.listdir(args.txt_files_dir) if fname.endswith('.txt')]
    
    if len(document_files) == 0:
        print(f"ERROR: No txt files found in {args.txt_files_dir}!")
        sys.exit(1)
        
    documents = []
    document_ids = []
    
    for file_path in tqdm(document_files, desc="Loading documents"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
                if document_text.strip():  # Only add non-empty documents
                    documents.append(document_text)
                    document_ids.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    if len(documents) == 0:
        print("ERROR: No valid documents loaded!")
        sys.exit(1)
        
    # Add documents to the RAG system
    rag_system.add_documents(documents, document_ids)
    
    # Load test questions
    test_questions_path = os.path.join(args.test_dir, args.questions_file)
    test_questions = load_questions(test_questions_path)
    
    if len(test_questions) == 0:
        print(f"ERROR: No questions loaded from {test_questions_path}!")
        sys.exit(1)
    
    # Extract question numbers
    question_numbers = [str(i + 1) for i in range(len(test_questions))]
    
    # Run the evaluation
    print(f"Running evaluation on {len(test_questions)} test questions...")
    results = {}
    
    for i, question in enumerate(tqdm(test_questions, desc="Answering questions")):
        question_idx = question_numbers[i]
        answer, _ = rag_system.answer_question(question, top_k=args.top_k)
        results[question_idx] = answer
    
    # Save results
    output_path = os.path.join(args.output_dir, f'system_output_{args.system_output_num}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()