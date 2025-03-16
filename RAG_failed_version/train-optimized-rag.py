import os
import json
import logging
import torch
import numpy as np
import random
from typing import List, Dict
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
        """Initialize RAG system"""
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.knowledge_base_dir = knowledge_base_dir
        self.vectorstore_dir = vectorstore_dir
        self.document_sources = []
        self.query_cache = {}
        
        # Training-derived improvements
        self.source_relevance = {}  # Source relevance scores for questions
        self.trained_prompts = {}   # Question-type to optimized prompt mapping
        self.keyword_mappings = {}  # Important keywords to sources mapping
        
        # Load embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs={"device": self.device_type},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Init text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )

        # Init LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Init pipeline
        self.reader = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.3
        )
        
        # Base prompt template
        self.base_prompt = """You are an AI assistant providing informative answers based on the given context. Use ANY relevant information from the context, even if it's only indirectly related. Provide your best guess rather than saying you don't know.

Context:
{context}

Question: {question}
Helpful answer:"""

    def _load_documents(self) -> List[Document]:
        """Load documents with source tracking"""
        documents = []
        all_sources = []
        
        for filename in os.listdir(self.knowledge_base_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.knowledge_base_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        all_sources.append(filename)
                        docs = self.text_splitter.create_documents(
                            [text],
                            metadatas=[{"source": filename}]
                        )
                        documents.extend(docs)
                except Exception as e:
                    logger.warning(f"File error {filename}: {str(e)}")
        
        # Store document sources for sampling
        self.document_sources = all_sources
        return documents

    def build_vectorstore(self):
        """Build vector database"""
        try:
            if not os.path.exists(self.vectorstore_dir):
                os.makedirs(self.vectorstore_dir, exist_ok=True)

            documents = self._load_documents()
            if not documents:
                raise ValueError("No valid documents loaded")

            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder,
                persist_directory=self.vectorstore_dir
            )
            self.vectorstore.persist()
        except Exception as e:
            logger.error(f"Vector database construction failed: {str(e)}")
            raise

    def learn_from_training_data(self, training_data_path: str):
        """Learn from training data to optimize retrieval and generation"""
        try:
            if not os.path.exists(training_data_path):
                logger.error(f"Training data not found: {training_data_path}")
                return
                
            # Load training data
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
                
            logger.info(f"Learning from {len(training_data)} training examples")
            
            # Initialize source relevance counters
            source_question_counter = {}  # How many times source X was relevant for questions
            total_questions = len(training_data)
            
            # Process each training example
            for example in training_data:
                question = example.get("question", "")
                answer = example.get("answer", "")
                source = example.get("source_file", "")
                
                # Skip if missing data
                if not question or not answer:
                    continue
                    
                # Extract keywords
                keywords = self._extract_keywords(question)
                
                # Update keyword to source mappings
                for keyword in keywords:
                    if keyword not in self.keyword_mappings:
                        self.keyword_mappings[keyword] = []
                    if source and source not in self.keyword_mappings[keyword]:
                        self.keyword_mappings[keyword].append(source)
                
                # Update source relevance counter
                if source:
                    source_question_counter[source] = source_question_counter.get(source, 0) + 1
                    
                # Create question type mappings based on question words
                q_type = self._determine_question_type(question)
                if q_type and q_type not in self.trained_prompts:
                    # Create specialized prompts for question types
                    if "when" in q_type or "year" in q_type or "date" in q_type:
                        self.trained_prompts[q_type] = """You are answering a date/time question. Look carefully for dates, years, or time periods in the context. If the exact date isn't mentioned, provide your best estimate based on available information.

Context:
{context}

Question: {question}
Helpful answer:"""
                    elif "who" in q_type or "person" in q_type:
                        self.trained_prompts[q_type] = """You are answering a question about a person or organization. Look carefully for names, titles, roles, and affiliations in the context. If the exact person isn't mentioned, provide your best guess based on available information.

Context:
{context}

Question: {question}
Helpful answer:"""
                    elif "where" in q_type or "location" in q_type:
                        self.trained_prompts[q_type] = """You are answering a location question. Look carefully for place names, geographic references, buildings, or addresses in the context. If the exact location isn't mentioned, provide your best guess based on available information.

Context:
{context}

Question: {question}
Helpful answer:"""
            
            # Calculate source relevance scores (normalized)
            total_counts = sum(source_question_counter.values())
            if total_counts > 0:
                for source, count in source_question_counter.items():
                    self.source_relevance[source] = count / total_counts
                    
            logger.info(f"Learned keyword mappings for {len(self.keyword_mappings)} keywords")
            logger.info(f"Created specialized prompts for {len(self.trained_prompts)} question types")
            logger.info(f"Calculated relevance scores for {len(self.source_relevance)} sources")
                
        except Exception as e:
            logger.error(f"Training data processing error: {str(e)}")

    def _determine_question_type(self, question: str) -> str:
        """Determine question type based on question words"""
        question = question.lower()
        
        if any(w in question for w in ["when", "year", "date", "time", "period", "century", "decade"]):
            return "temporal"
        elif any(w in question for w in ["who", "person", "name", "individual", "professor", "student"]):
            return "person"
        elif any(w in question for w in ["where", "location", "place", "building", "city", "campus"]):
            return "location"
        elif any(w in question for w in ["what", "which", "how"]):
            return "factual"
        elif any(w in question for w in ["why", "reason", "cause", "effect", "impact"]):
            return "explanatory"
        else:
            return "general"

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", 
                      "in", "on", "at", "to", "for", "with", "by", "about", "like", "of"}
        
        words = query.lower().replace("?", "").replace(".", "").replace(",", "").split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Add 2-gram and 3-gram phrases
        phrases = []
        if len(words) >= 2:
            phrases.extend([" ".join(words[i:i+2]) for i in range(len(words)-1)])
        if len(words) >= 3:
            phrases.extend([" ".join(words[i:i+3]) for i in range(len(words)-2)])
            
        return keywords + phrases

    def retrieve_context(self, query: str) -> List[Dict]:
        """Optimized retrieval using training-derived knowledge"""
        if query in self.query_cache:
            return self.query_cache[query]
            
        try:
            results = []
            
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Get relevant sources based on keywords
            relevant_sources = set()
            for keyword in keywords:
                if keyword in self.keyword_mappings:
                    relevant_sources.update(self.keyword_mappings[keyword])
            
            # 1. Get vector search results (fast)
            if hasattr(self, "vectorstore"):
                try:
                    # Try with original query
                    vector_results = self.vectorstore.similarity_search(query, k=15)
                    results.extend(vector_results)
                    
                    # Add results from relevant sources if available
                    if relevant_sources:
                        for source in relevant_sources:
                            source_results = self.vectorstore.similarity_search(
                                query, 
                                k=5,
                                filter={"source": source}
                            )
                            results.extend(source_results)
                except Exception as e:
                    logger.warning(f"Vector search failed: {str(e)}")
            
            # 2. Add random samples from each source document (covers all files)
            if self.document_sources:
                sample_results = self._sample_from_sources(query)
                results.extend(sample_results)
            
            # Deduplicate
            unique_docs = {}
            for doc in results:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
            
            # Format results
            formatted = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown")
                }
                for doc in unique_docs.values()
            ]
            
            # Cache and return
            self.query_cache[query] = formatted
            return formatted
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    def _sample_from_sources(self, query: str, max_samples=3) -> List[Document]:
        """Get samples from each source document, prioritizing relevant sources"""
        if not self.document_sources:
            if not os.path.exists(self.knowledge_base_dir):
                return []
            self._load_documents()
            if not self.document_sources:
                return []
        
        results = []
        
        # Determine source priority based on training data
        sorted_sources = sorted(
            self.document_sources,
            key=lambda s: self.source_relevance.get(s, 0),
            reverse=True
        )
        
        # Sample from each source file
        for source in sorted_sources:
            try:
                # Get documents for this source
                docs = self.vectorstore.similarity_search(
                    query, 
                    k=max_samples,
                    filter={"source": source}
                )
                
                # If no results with filter, try fetching any document from this source
                if not docs:
                    docs = self.vectorstore.similarity_search(
                        source,  # Use source name as query
                        k=max_samples
                    )
                
                # Add up to max_samples from this source
                results.extend(docs[:max_samples])
                
            except Exception as e:
                logger.warning(f"Error sampling from {source}: {str(e)}")
                
        return results

    def generate_answer(self, question: str) -> str:
        """Generate answer with training-optimized prompts"""
        try:
            context_docs = self.retrieve_context(question)
            
            if context_docs:
                # Limit context to avoid token limits
                if len(context_docs) > 15:
                    # Keep some top results and some random ones
                    top_docs = context_docs[:10]
                    random_docs = random.sample(context_docs[10:], min(5, len(context_docs)-10))
                    context_docs = top_docs + random_docs
                
                context = "\n\n".join([
                    f"[Source: {doc['source']}]\n{doc['content']}" 
                    for doc in context_docs
                ])
                
                # Select appropriate prompt based on question type
                q_type = self._determine_question_type(question)
                prompt_template = self.trained_prompts.get(q_type, self.base_prompt)
                
                prompt = prompt_template.format(
                    context=context,
                    question=question
                )

                response = self.reader(prompt)[0]['generated_text']
                
                # Extract answer
                answer_marker = "Helpful answer:"
                if answer_marker in response:
                    answer = response.split(answer_marker)[-1].strip()
                    
                    # Clean up repetitive content
                    if "I don't have" in answer and len(answer) > 100:
                        return "I don't have specific information about that."
                    
                    return self._clean_repetitive_text(answer)
                else:
                    # Fallback extraction
                    question_marker = f"Question: {question}"
                    if question_marker in response:
                        answer = response.split(question_marker)[-1].strip()
                        if "Helpful answer:" in answer:
                            answer = answer.split("Helpful answer:")[-1].strip()
                        return answer
                    
                    return "Based on available information, I can provide a partial answer."
            else:
                return "I don't have specific information on this topic."
                
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I encountered an issue retrieving information on this topic."

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
                
                # Final cleaning
                if "Cannot answer based on available knowledge" in answer:
                    answer = "Based on available information, I can provide a partial answer."
                
                results[str(idx)] = answer

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Test set processing failed: {str(e)}")
            raise

if __name__ == "__main__":
    # ================== Configuration Section ==================
    TXT_FOLDER = "/gpfsnyu/scratch/yx2432/Research/Zhuzi/Project2/llama3_try/txt_files"
    TRAIN_DATA = "/gpfsnyu/scratch/yx2432/Research/Zhuzi/Project2/llama3_try/train_test_split/train/full_qa_pairs.json"
    TEST_QUESTIONS = "/gpfsnyu/scratch/yx2432/Research/Zhuzi/Project2/llama3_try/train_test_split/test/questions.txt"
    OUTPUT_FILE = "train_optimized_output.json"
    VECTOR_DB_PATH = "train_optimized_db"
    # ===========================================================

    try:
        # Initialize system
        rag = RAGSystem(
            knowledge_base_dir=TXT_FOLDER,
            vectorstore_dir=VECTOR_DB_PATH
        )

        # Build vector database (required for first run)
        if not os.path.exists(VECTOR_DB_PATH) or len(os.listdir(VECTOR_DB_PATH)) == 0:
            logger.info("Building vector database...")
            rag.build_vectorstore()
            
        # Learn from training data
        logger.info("Learning from training data...")
        rag.learn_from_training_data(TRAIN_DATA)

        # Process test set
        logger.info("Processing test questions...")
        rag.process_test_set(
            test_questions_path=TEST_QUESTIONS,
            output_file=OUTPUT_FILE
        )

    except Exception as e:
        logger.critical(f"System runtime error: {str(e)}")
        exit(1)