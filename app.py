import os
import shutil
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from pinecone import Pinecone, PineconeException
import PyPDF2
import docx
import uuid
import time
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY =os.getenv('PINECONE_API_KEY')


INDEX_NAME = "customer-hub"


# Get API keys from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

# Initialize clients
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY.strip())
    pc = Pinecone(api_key=PINECONE_API_KEY.strip())
    
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY.strip(),
        model="text-embedding-ada-002"
    )
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            index = pc.Index(INDEX_NAME)
            break
        except PineconeException as pe:
            if attempt == max_retries - 1:
                logger.error(f"Failed to initialize Pinecone after {max_retries} attempts: {str(pe)}")
                raise
            time.sleep(retry_delay)
            retry_delay *= 2
            
except Exception as e:
    logger.error(f"Error initializing clients: {str(e)}")
    raise

app = FastAPI(
    title="Document QA API",
    description="API for document processing and question answering",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

class DocumentProcessor:
    def __init__(self):
        # Initialize semantic chunker with standard deviation threshold
        self.text_splitter = SemanticChunker(
            embeddings_model,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=3.0,  # Number of standard deviations
            min_chunk_size=200  # Minimum characters per chunk
        )

    def extract_text(self, file_path: str, file_extension: str) -> str:
        try:
            if file_extension == 'pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension in ['doc', 'docx']:
                return self._extract_from_docx(file_path)
            else:
                return self._extract_from_txt(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    def _extract_from_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _extract_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()

    async def process_document(self, file_path: str, filename: str, file_extension: str) -> List[dict]:
        text = self.extract_text(file_path, file_extension)
        
        # Create documents using semantic chunking
        documents = self.text_splitter.create_documents([text])
        
        vectors = []
        for i, doc in enumerate(documents):
            try:
                embedding = embeddings_model.embed_query(doc.page_content)
                chunk_id = f"{filename}-{i}-{uuid.uuid4()}"
                
                vector = {
                    'id': chunk_id,
                    'values': embedding,
                    'metadata': {
                        'text': doc.page_content,
                        'filename': filename,
                        'timestamp': time.time()
                    }
                }
                vectors.append(vector)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i} of {filename}: {str(e)}")
                continue
                
        return vectors
    
class QueryProcessor:
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold

    async def process_query(self, query: str) -> Dict[str, Any]:
        try:
            query_embedding = embeddings_model.embed_query(query)
            
            search_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            if not search_results['matches']:
                return {
                    "response": "No documents found. Please upload documents first.",
                    "source": None,
                    "similarity_score": 0
                }
            
            matching_results = [
                match for match in search_results['matches'] 
                if match['score'] >= self.similarity_threshold
            ]
            
            if not matching_results:
                return {
                    "response": "No relevant information found. Please try rephrasing your question.",
                    "source": None,
                    "similarity_score": 0
                }
            
            top_match = matching_results[0]
            context = top_match['metadata'].get('text', '')
            
            response = await self._generate_response(query, context)
            
            return {
                "response": response,
                "source": top_match['metadata'].get('filename', 'unknown'),
                "similarity_score": round(top_match['score'], 3)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    async def _generate_response(self, query: str, context: str) -> str:
        prompt = f"""Based on the following context, please provide a brief and direct answer to the question.
        If the context doesn't contain relevant information, say so.

        Question: {query}

        Context: {context}

        Answer:"""
        
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant. Provide brief, direct answers based on the given context. If information is not available in the context, say so clearly."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        return completion.choices[0].message.content

class ChatSession:
    def __init__(self):
        self.messages = []
        self.last_interaction = time.time()

class ChatProcessor:
    def __init__(self):
        self.sessions = {}
    
    def get_or_create_session(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession()
        return self.sessions[session_id]
    
    async def process_message(self, message: str, session_id: str) -> Dict[str, Any]:
        session = self.get_or_create_session(session_id)
        session.messages.append({"role": "user", "content": message})
        system_prompt = """
        You are a chatbot for 'The New ToolBox', a company parented by company named 'We Build Trades' Owned by Daniel J Brown. 
        - For Generic Quries, be professional and authentic
        - Always talk about 'The New Tool Box', as you are acting the chatbot of this company.
        - The New ToolBox is Comapny deals with client who mostly deals with Boilers and solar panels and related products

        Be professional, helpful, and knowledgeable about our tools and services.
        Anwer the 
        """
        # Prepare conversation history
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history
        messages.extend(session.messages[-5:])  # Keep last 5 messages for context
        
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            response = completion.choices[0].message.content
            session.messages.append({"role": "assistant", "content": response})
            
            return {
                "response": response,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            raise

# Initialize processors
chat_processor = ChatProcessor()

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    processor = DocumentProcessor()
    uploaded_files = []
    
    try:
        for file in files:
            file_extension = file.filename.split('.')[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                logger.warning(f"Skipping file with unsupported extension: {file.filename}")
                continue
            
            temp_file_path = f"temp_{uuid.uuid4()}.{file_extension}"
            try:
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                vectors = await processor.process_document(
                    temp_file_path,
                    file.filename,
                    file_extension
                )
                
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    index.upsert(vectors=batch)
                    time.sleep(0.5)  # Rate limiting
                
                uploaded_files.append(file.filename)
                
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
        return JSONResponse(
            content={
                "message": "Successfully processed files",
                "uploaded_files": uploaded_files
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_documents(query: str = Body(..., embed=True)):
    try:
        query_processor = QueryProcessor(similarity_threshold=0.5)
        result = await query_processor.process_query(query)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(message: str = Body(...), session_id: str = Body(...)):
    try:
        result = await chat_processor.process_message(message, session_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=6969, reload=True)