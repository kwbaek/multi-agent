from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from typing import List, Optional
import os
from config import settings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self.vector_store = None
        self.init_vector_store()
    
    def init_vector_store(self):
        """벡터 스토어를 초기화합니다."""
        try:
            # 기존 벡터 스토어가 있다면 로드, 없다면 새로 생성
            if os.path.exists(settings.chroma_persist_directory):
                self.vector_store = Chroma(
                    persist_directory=settings.chroma_persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                self.vector_store = Chroma(
                    persist_directory=settings.chroma_persist_directory,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            print(f"벡터 스토어 초기화 실패: {e}")
            # 새로운 벡터 스토어 생성
            self.vector_store = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=self.embeddings
            )
    
    def load_and_split_pdf(self, file_path: str) -> List[Document]:
        """PDF 파일을 로드하고 청크로 분할합니다."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_and_split_text(self, file_path: str) -> List[Document]:
        """텍스트 파일을 로드하고 청크로 분할합니다."""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """문서들을 벡터 스토어에 추가합니다."""
        if not self.vector_store:
            self.init_vector_store()
        
        doc_ids = self.vector_store.add_documents(documents)
        self.vector_store.persist()
        return doc_ids
    
    def add_text(self, text: str, metadata: Optional[dict] = None) -> str:
        """텍스트를 문서로 변환하여 벡터 스토어에 추가합니다."""
        document = Document(page_content=text, metadata=metadata or {})
        split_docs = self.text_splitter.split_documents([document])
        return self.add_documents(split_docs)[0]
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """유사도 검색을 수행합니다."""
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """점수와 함께 유사도 검색을 수행합니다."""
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def delete_collection(self):
        """벡터 스토어 컬렉션을 삭제합니다."""
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """검색기를 반환합니다."""
        if not self.vector_store:
            self.init_vector_store()
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

# 전역 벡터 스토어 매니저 인스턴스
vector_store_manager = VectorStoreManager()