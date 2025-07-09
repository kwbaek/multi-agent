from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import shutil
from pathlib import Path

from config import settings
from vector_store import vector_store_manager
from rag_workflow import rag_workflow, enhanced_rag_workflow

app = FastAPI(
    title="MD-Agent RAG API",
    description="LangGraph 기반 RAG 시스템 API",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서만 사용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델 정의
class QueryRequest(BaseModel):
    question: str
    use_enhanced: bool = True

class QueryResponse(BaseModel):
    query: str
    answer: str
    classification: Optional[str] = None
    retrieved_docs: List[dict]
    context: str

class TextUploadRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None

class UploadResponse(BaseModel):
    message: str
    doc_ids: List[str]

@app.get("/")
def read_root():
    return {
        "message": "MD-Agent RAG 시스템이 실행중입니다",
        "version": "1.0.0",
        "features": [
            "문서 업로드 (PDF, 텍스트)",
            "LangGraph 기반 RAG 워크플로우",
            "향상된 질문 분류 시스템",
            "벡터 검색 및 생성"
        ]
    }

@app.post("/upload/file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """PDF 또는 텍스트 파일을 업로드하고 벡터 스토어에 추가합니다."""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 없습니다")
    
    # 지원되는 파일 형식 확인
    supported_extensions = ['.pdf', '.txt']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"지원되지 않는 파일 형식입니다. 지원 형식: {supported_extensions}"
        )
    
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        # 파일 형식에 따라 처리
        if file_extension == '.pdf':
            documents = vector_store_manager.load_and_split_pdf(tmp_file_path)
        else:  # .txt
            documents = vector_store_manager.load_and_split_text(tmp_file_path)
        
        # 문서를 벡터 스토어에 추가
        doc_ids = vector_store_manager.add_documents(documents)
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        return UploadResponse(
            message=f"{file.filename} 파일이 성공적으로 업로드되었습니다. {len(documents)}개의 청크로 분할되었습니다.",
            doc_ids=doc_ids
        )
        
    except Exception as e:
        # 임시 파일 정리
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/upload/text", response_model=UploadResponse)
async def upload_text(request: TextUploadRequest):
    """텍스트를 직접 업로드하고 벡터 스토어에 추가합니다."""
    
    try:
        doc_id = vector_store_manager.add_text(request.text, request.metadata)
        
        return UploadResponse(
            message="텍스트가 성공적으로 업로드되었습니다.",
            doc_ids=[doc_id]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """RAG 시스템에 질문하고 답변을 받습니다."""
    
    try:
        if request.use_enhanced:
            result = enhanced_rag_workflow.run_rag(request.question)
        else:
            result = rag_workflow.run_rag(request.question)
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 처리 중 오류가 발생했습니다: {str(e)}")

@app.get("/search/{query}")
async def search_documents(query: str, k: int = 4):
    """벡터 스토어에서 문서를 검색합니다 (점수와 함께)."""
    
    try:
        results = vector_store_manager.similarity_search_with_score(query, k=k)
        
        return {
            "query": query,
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")

@app.get("/health")
async def health_check():
    """시스템 상태를 확인합니다."""
    
    try:
        # 벡터 스토어 상태 확인
        vector_store_status = "OK" if vector_store_manager.vector_store else "ERROR"
        
        # OpenAI API 키 확인
        api_key_status = "OK" if settings.openai_api_key else "MISSING"
        
        return {
            "status": "healthy",
            "vector_store": vector_store_status,
            "openai_api_key": api_key_status,
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 확인 중 오류가 발생했습니다: {str(e)}")

@app.delete("/vector-store")
async def clear_vector_store():
    """벡터 스토어의 모든 데이터를 삭제합니다."""
    
    try:
        vector_store_manager.delete_collection()
        vector_store_manager.init_vector_store()
        
        return {"message": "벡터 스토어가 성공적으로 초기화되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 스토어 초기화 중 오류가 발생했습니다: {str(e)}")

@app.get("/info")
async def get_system_info():
    """시스템 정보를 반환합니다."""
    
    return {
        "system": "MD-Agent RAG",
        "description": "LangGraph를 사용한 고급 RAG 시스템",
        "features": {
            "workflow_engine": "LangGraph",
            "vector_store": "ChromaDB",
            "embeddings": settings.embedding_model,
            "llm": settings.llm_model,
            "supported_formats": ["PDF", "TXT"],
            "question_classification": True,
            "korean_support": True
        },
        "endpoints": {
            "upload_file": "POST /upload/file",
            "upload_text": "POST /upload/text", 
            "query": "POST /query",
            "search": "GET /search/{query}",
            "health": "GET /health",
            "clear": "DELETE /vector-store",
            "info": "GET /info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
