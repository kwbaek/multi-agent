"""
LangGraph RAG 시스템 사용 예제

이 스크립트는 RAG 시스템을 프로그래밍 방식으로 사용하는 방법을 보여줍니다.
"""

import asyncio
import os
from vector_store import vector_store_manager
from rag_workflow import rag_workflow, enhanced_rag_workflow

async def example_usage():
    """RAG 시스템 사용 예제"""
    
    print("🚀 LangGraph RAG 시스템 예제 시작")
    print("=" * 50)
    
    # 1. 샘플 텍스트 데이터 추가
    print("\n📄 샘플 문서 추가 중...")
    
    sample_documents = [
        {
            "text": """
            LangGraph는 LangChain에서 개발한 라이브러리로, 복잡한 멀티 에이전트 워크플로우를 
            그래프 형태로 구성할 수 있게 해줍니다. 이를 통해 조건부 라우팅, 루프, 병렬 처리 등의 
            고급 기능을 쉽게 구현할 수 있습니다.
            """,
            "metadata": {"source": "langgraph_intro", "type": "definition"}
        },
        {
            "text": """
            RAG(Retrieval-Augmented Generation)는 정보 검색과 생성형 AI를 결합한 기술입니다. 
            먼저 관련 문서를 검색하고, 그 정보를 바탕으로 정확하고 맥락에 맞는 답변을 생성합니다. 
            이는 AI 모델이 최신 정보나 특정 도메인 지식에 접근할 수 있게 해줍니다.
            """,
            "metadata": {"source": "rag_explanation", "type": "definition"}
        },
        {
            "text": """
            벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색할 수 있는 데이터베이스입니다. 
            ChromaDB는 오픈소스 벡터 데이터베이스로, 임베딩 벡터의 유사도 검색을 빠르게 수행할 수 있습니다. 
            RAG 시스템에서 문서 검색 엔진의 역할을 담당합니다.
            """,
            "metadata": {"source": "vector_db_info", "type": "technical"}
        }
    ]
    
    for i, doc in enumerate(sample_documents):
        doc_id = vector_store_manager.add_text(doc["text"], doc["metadata"])
        print(f"   ✅ 문서 {i+1} 추가 완료 (ID: {doc_id[:8]}...)")
    
    # 2. 기본 RAG 워크플로우 테스트
    print("\n🔍 기본 RAG 워크플로우 테스트")
    print("-" * 30)
    
    queries = [
        "LangGraph가 무엇인가요?",
        "RAG 기술에 대해 설명해주세요",
        "벡터 데이터베이스의 역할은 무엇인가요?"
    ]
    
    for query in queries:
        print(f"\n질문: {query}")
        result = rag_workflow.run_rag(query)
        print(f"답변: {result['answer']}")
        print(f"검색된 문서 수: {len(result['retrieved_docs'])}")
    
    # 3. 향상된 RAG 워크플로우 테스트 (질문 분류 포함)
    print("\n🎯 향상된 RAG 워크플로우 테스트 (질문 분류)")
    print("-" * 40)
    
    enhanced_queries = [
        "안녕하세요!",  # 직접 답변 예상
        "ChromaDB에 대해 알려주세요",  # 검색 기반 답변 예상
        "오늘 날씨는 어때요?",  # 직접 답변 예상
        "LangGraph의 주요 기능은 무엇인가요?"  # 검색 기반 답변 예상
    ]
    
    for query in enhanced_queries:
        print(f"\n질문: {query}")
        result = enhanced_rag_workflow.run_rag(query)
        print(f"분류: {result['classification']}")
        print(f"답변: {result['answer']}")
        if result['retrieved_docs']:
            print(f"검색된 문서 수: {len(result['retrieved_docs'])}")
        else:
            print("검색된 문서 없음 (직접 답변)")
    
    # 4. 벡터 검색 테스트
    print("\n🔎 벡터 검색 테스트")
    print("-" * 20)
    
    search_query = "AI 기술"
    search_results = vector_store_manager.similarity_search_with_score(search_query, k=3)
    
    print(f"검색어: {search_query}")
    for i, (doc, score) in enumerate(search_results):
        print(f"\n결과 {i+1} (점수: {score:.4f}):")
        print(f"내용: {doc.page_content[:100]}...")
        print(f"메타데이터: {doc.metadata}")
    
    print("\n✨ 예제 완료!")
    print("API 서버를 시작하려면 'uvicorn main:app --reload' 명령을 실행하세요.")

def sync_example():
    """동기 버전 예제 (OpenAI API 키가 없을 때를 위한)"""
    
    print("🚀 LangGraph RAG 시스템 예제 (동기 버전)")
    print("=" * 50)
    
    # OpenAI API 키 확인
    from config import settings
    if not settings.openai_api_key:
        print("⚠️  OpenAI API 키가 설정되지 않았습니다.")
        print("환경 변수 OPENAI_API_KEY를 설정하거나 .env 파일을 생성하세요.")
        print("\n예시:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\n또는 .env 파일에:")
        print("OPENAI_API_KEY=your-api-key-here")
        return
    
    # 1. 샘플 데이터 추가
    print("\n📄 샘플 문서 추가 중...")
    
    sample_text = """
    머신러닝은 인공지능의 한 분야로, 컴퓨터가 명시적으로 프로그래밍되지 않고도 
    데이터로부터 학습할 수 있게 하는 기술입니다. 딥러닝은 머신러닝의 하위 분야로, 
    인공 신경망을 사용하여 복잡한 패턴을 학습합니다.
    """
    
    try:
        doc_id = vector_store_manager.add_text(sample_text, {"source": "ml_basic"})
        print(f"   ✅ 문서 추가 완료 (ID: {doc_id[:8]}...)")
    except Exception as e:
        print(f"   ❌ 문서 추가 실패: {e}")
        return
    
    # 2. 간단한 검색 테스트
    print("\n🔍 벡터 검색 테스트")
    
    try:
        search_results = vector_store_manager.similarity_search("머신러닝이란", k=2)
        
        for i, doc in enumerate(search_results):
            print(f"\n결과 {i+1}:")
            print(f"내용: {doc.page_content[:100]}...")
            print(f"메타데이터: {doc.metadata}")
            
    except Exception as e:
        print(f"   ❌ 검색 실패: {e}")

if __name__ == "__main__":
    # 환경 설정 확인
    from config import settings
    
    if settings.openai_api_key:
        # OpenAI API 키가 있으면 전체 예제 실행
        try:
            import asyncio
            asyncio.run(example_usage())
        except Exception as e:
            print(f"비동기 예제 실행 실패: {e}")
            print("동기 버전으로 실행 중...")
            sync_example()
    else:
        # API 키가 없으면 동기 버전 실행
        sync_example()