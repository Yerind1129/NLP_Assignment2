import os
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, 
                 knowledge_base_dir: str,
                 vectorstore_dir: str = "chroma_db",
                 embedder_name: str = "/gpfsnyu/scratch/yx2432/models/BAAI-bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09",
                 llm_path: str = "/gpfsnyu/scratch/yx2432/models/llama-3.1-8b-instruct"):
        """
        Initialize RAG system
        :param knowledge_base_dir: Path to the raw text files directory
        :param vectorstore_dir: Path to store the vector database
        :param embedder_name: Name of the embedding model
        :param llm_path: Path to the large language model
        """
        # Device detection and logging
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing RAG system | Device: {self.device_type.upper()} | Model: {os.path.basename(llm_path)}")

        # Validate knowledge base path
        if not os.path.isdir(knowledge_base_dir):
            raise FileNotFoundError(f"Knowledge base path not found: {knowledge_base_dir}")
        self.knowledge_base_dir = knowledge_base_dir
        self.vectorstore_dir = vectorstore_dir
        
        # Store all document chunks for brute force search
        self.all_documents = []
        
        # Cache for query results
        self.query_cache = {}

        # Initialize embedding model
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs={"device": self.device_type},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Initialize text splitter - smaller chunk size for better matching
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=150,
            length_function=len
        )

        # Initialize large language model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
            
            # Avoid quantization, use float32
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

        # Initialize generation pipeline with settings to reduce repetition
        self.reader = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200, 
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )
        
        # Store optimized prompts
        self.prompt_templates = {
            "default": """You are an AI assistant providing informative answers based on the given context and general knowledge. Use ANY relevant information from the context, even if it's only indirectly related to the question. Provide your best guess based on available information rather than saying you don't know. If nothing in the context is relevant, briefly state you don't have that specific information.

Context:
{context}

Question: {question}
Helpful answer:"""
        }

    def _load_documents(self) -> List[Document]:
        """Load and split all text files"""
        documents = []
        file_count = 0
        for filename in os.listdir(self.knowledge_base_dir):
            if filename.endswith(".txt"):
                file_count += 1
                file_path = os.path.join(self.knowledge_base_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        docs = self.text_splitter.create_documents(
                            [text],
                            metadatas=[{"source": filename}]
                        )
                        documents.extend(docs)
                        logger.debug(f"Loaded file: {filename} | Chunks: {len(docs)}")
                except UnicodeDecodeError:
                    logger.warning(f"File decoding failed: {filename}")
                except Exception as e:
                    logger.error(f"File loading error: {filename} | Error: {str(e)}")
        
        logger.info(f"Loaded {file_count} text files, resulting in {len(documents)} document chunks")
        # Store all documents for brute force search
        self.all_documents = documents
        return documents

    def build_vectorstore(self):
        """Build vector database"""
        try:
            if not os.path.exists(self.vectorstore_dir):
                os.makedirs(self.vectorstore_dir, exist_ok=True)

            documents = self._load_documents()
            if not documents:
                raise ValueError("No valid documents loaded")

            # Correct initialization
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder,
                persist_directory=self.vectorstore_dir
            )
            self.vectorstore.persist()

            logger.info(f"Vector database built successfully | Document chunks: {len(documents)} | Storage path: {self.vectorstore_dir}")
        except Exception as e:
            logger.error(f"Vector database construction failed: {str(e)}")
            raise

    def _brute_force_search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Perform brute force search through all documents"""
        if not self.all_documents:
            self._load_documents()
            if not self.all_documents:
                return []
        
        # Get query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Calculate similarities for all documents
        results = []
        for doc in self.all_documents:
            try:
                # Get document embedding
                doc_embedding = self.embedder.embed_query(doc.page_content)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                results.append({
                    "document": doc,
                    "similarity": similarity
                })
            except Exception as e:
                logger.error(f"Error calculating similarity: {str(e)}")
        
        # Sort by similarity and take top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Convert to standard format
        formatted_results = [
            {
                "content": r["document"].page_content,
                "source": r["document"].metadata.get("source", "unknown"),
                "similarity": r["similarity"]
            }
            for r in results[:max_results]
        ]
        
        return formatted_results

    def retrieve_context(self, query: str) -> List[Dict]:
        """Enhanced retrieval with vector search and brute force backup"""
        try:
            cache_key = query
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            # First try standard vector search
            vector_results = []
            if hasattr(self, "vectorstore"):
                try:
                    # Create query variations
                    query_variations = [
                        query,
                        f"{query} Pittsburgh", 
                        f"{query} Carnegie Mellon University"
                    ]
                    
                    # Get results for each query variation
                    for q_var in query_variations:
                        results = self.vectorstore.similarity_search(q_var, k=20)
                        vector_results.extend(results)
                except Exception as e:
                    logger.warning(f"Vector search failed: {str(e)}")
            
            # Convert vector results to standard format
            vector_formatted = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown")
                }
                for doc in vector_results
            ]
            
            # Backup: brute force search through all documents
            brute_force_results = self._brute_force_search(query, max_results=100)
            
            # Combine results and deduplicate
            all_results = vector_formatted + brute_force_results
            seen_content = set()
            unique_results = []
            
            for doc in all_results:
                if doc["content"] not in seen_content:
                    seen_content.add(doc["content"])
                    unique_results.append(doc)
            
            # Sort by similarity if available
            unique_results.sort(
                key=lambda x: x.get("similarity", 0),
                reverse=True
            )
            
            # Cache and return
            self.query_cache[cache_key] = unique_results[:50]  # Keep top 50
            return self.query_cache[cache_key]
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            # Fallback to brute force
            return self._brute_force_search(query)

    def generate_answer(self, question: str, prompt_key: str = "default") -> str:
        """Generate answer with controlled output format"""
        try:
            context_docs = self.retrieve_context(question)
            
            if context_docs:
                # Use a subset of documents to avoid token limits
                # Include both high-ranked and some random ones for coverage
                top_docs = context_docs[:10]
                
                # If we have more than 30 documents, also add some random ones
                if len(context_docs) > 30:
                    random_indices = np.random.choice(
                        range(15, len(context_docs)), 
                        size=min(5, len(context_docs)-15), 
                        replace=False
                    )
                    for i in random_indices:
                        top_docs.append(context_docs[i])
                
                context = "\n\n".join([
                    f"[Source: {doc['source']}]\n{doc['content']}" 
                    for doc in top_docs
                ])
                
                prompt = self.prompt_templates[prompt_key].format(
                    context=context,
                    question=question
                )

                response = self.reader(prompt)[0]['generated_text']
                
                # Extract answer
                answer_marker = "Helpful answer:"
                if answer_marker in response:
                    answer = response.split(answer_marker)[-1].strip()
                    
                    # Clean up repetitive content
                    if answer.count("I don't have") > 1:
                        return "I don't have specific information about that."
                    
                    # Check and clean repetitive text
                    if self._is_repetitive(answer):
                        return self._clean_repetitive_text(answer)
                    
                    return answer
                else:
                    # Fallback extraction
                    question_marker = f"Question: {question}"
                    if question_marker in response:
                        answer = response.split(question_marker)[-1].strip()
                        if "Helpful answer:" in answer:
                            answer = answer.split("Helpful answer:")[-1].strip()
                        return answer
                    
                    return "Based on the available information, I can provide a partial answer, but details may be limited."
            else:
                # If no context found
                return "While I don't have specific information on this topic, it may be related to Carnegie Mellon University or Pittsburgh."
                
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return f"I encountered an issue while retrieving information about this topic."

    def _is_repetitive(self, text: str) -> bool:
        """Check if text contains repetitive patterns"""
        if not text:
            return False
            
        # Check for repeated sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) >= 3:
            # Check for exactly repeated sentences
            unique_sentences = set(sentences)
            if len(unique_sentences) < len(sentences) * 0.7:  # More than 30% repetition
                return True
                
        # Check for repeated phrases (5+ words)
        words = text.split()
        if len(words) >= 15:  # Only check longer texts
            phrases = [' '.join(words[i:i+5]) for i in range(len(words)-5)]
            phrase_count = {}
            for phrase in phrases:
                phrase_count[phrase] = phrase_count.get(phrase, 0) + 1
                
            # If any 5-word phrase appears 3+ times, it's repetitive
            if any(count >= 3 for count in phrase_count.values()):
                return True
                
        return False
    
    def _clean_repetitive_text(self, text: str) -> str:
        """Clean repetitive text by removing duplicates"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return text
            
        # Keep only unique sentences in order
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
                
        # Rejoin with periods
        clean_text = '. '.join(unique_sentences)
        if not clean_text.endswith('.'):
            clean_text += '.'
            
        return clean_text
            
    def process_test_set(self, test_questions_path: str, output_file: str):
        """Process test question set"""
        try:
            with open(test_questions_path, "r", encoding="utf-8") as f:
                questions = [q.strip() for q in f.readlines() if q.strip()]

            results = {}
            for idx, question in enumerate(questions, 1):
                logger.info(f"Processing question {idx}/{len(questions)}: {question[:50]}...")
                answer = self.generate_answer(question)
                
                # Final cleaning for "Cannot answer" responses
                if "Cannot answer based on available knowledge" in answer:
                    answer = "Based on available information, I can only provide a partial answer to this question."
                
                results[str(idx)] = answer

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Test results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Test set processing failed: {str(e)}")
            raise

if __name__ == "__main__":
    # ================== Configuration Section ==================
    TXT_FOLDER = "/gpfsnyu/scratch/yx2432/Research/Zhuzi/Project2/llama3_try/txt_files"
    TEST_QUESTIONS = "/gpfsnyu/scratch/yx2432/Research/Zhuzi/Project2/llama3_try/train_test_split/test/test_set.txt"
    OUTPUT_FILE = "full_retrieval_output.json"
    VECTOR_DB_PATH = "full_retrieval_db"
    # ===========================================================

    try:
        # Environment check
        if torch.cuda.is_available():
            logger.info(f"GPU available | Model: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        else:
            logger.warning("No GPU detected, running in CPU mode")

        # Initialize system
        rag = RAGSystem(
            knowledge_base_dir=TXT_FOLDER,
            vectorstore_dir=VECTOR_DB_PATH
        )

        # Build vector database (required for first run)
        if not os.path.exists(VECTOR_DB_PATH) or len(os.listdir(VECTOR_DB_PATH)) == 0:
            logger.info("Building vector database...")
            rag.build_vectorstore()

        # Process test set
        logger.info("Processing test questions...")
        rag.process_test_set(
            test_questions_path=TEST_QUESTIONS,
            output_file=OUTPUT_FILE
        )

    except Exception as e:
        logger.critical(f"System runtime error: {str(e)}")
        exit(1)