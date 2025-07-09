"""
LangGraph RAG 시스템 테스트 스크립트

이 스크립트는 RAG 시스템의 주요 컴포넌트들이 올바르게 작동하는지 확인합니다.
"""

import os
import sys
import traceback

def test_imports():
    """필요한 모듈들이 제대로 import되는지 테스트"""
    print("📦 모듈 import 테스트...")
    
    try:
        from config import settings
        print("   ✅ config 모듈 import 성공")
    except Exception as e:
        print(f"   ❌ config 모듈 import 실패: {e}")
        return False
    
    try:
        from vector_store import VectorStoreManager
        print("   ✅ vector_store 모듈 import 성공")
    except Exception as e:
        print(f"   ❌ vector_store 모듈 import 실패: {e}")
        return False
    
    try:
        from rag_workflow import RAGWorkflow, EnhancedRAGWorkflow
        print("   ✅ rag_workflow 모듈 import 성공")
    except Exception as e:
        print(f"   ❌ rag_workflow 모듈 import 실패: {e}")
        return False
    
    return True

def test_configuration():
    """설정이 올바르게 로드되는지 테스트"""
    print("\n⚙️  설정 테스트...")
    
    try:
        from config import settings
        
        print(f"   OpenAI API Key: {'설정됨' if settings.openai_api_key else '미설정'}")
        print(f"   Chunk Size: {settings.chunk_size}")
        print(f"   Chunk Overlap: {settings.chunk_overlap}")
        print(f"   Embedding Model: {settings.embedding_model}")
        print(f"   LLM Model: {settings.llm_model}")
        
        if not settings.openai_api_key:
            print("   ⚠️  OpenAI API 키가 설정되지 않았습니다.")
            print("   환경 변수 OPENAI_API_KEY를 설정하거나 .env 파일을 생성하세요.")
            return False
        
        print("   ✅ 설정 로드 성공")
        return True
        
    except Exception as e:
        print(f"   ❌ 설정 로드 실패: {e}")
        return False

def test_vector_store():
    """벡터 스토어 기본 기능 테스트"""
    print("\n🗄️  벡터 스토어 테스트...")
    
    try:
        from vector_store import VectorStoreManager
        
        # 벡터 스토어 매니저 생성
        vsm = VectorStoreManager()
        print("   ✅ VectorStoreManager 초기화 성공")
        
        # 테스트 텍스트 추가
        test_text = "이것은 벡터 스토어 테스트를 위한 샘플 텍스트입니다."
        doc_id = vsm.add_text(test_text, {"test": True})
        print(f"   ✅ 텍스트 추가 성공 (ID: {doc_id[:8] if doc_id else 'None'}...)")
        
        # 검색 테스트
        results = vsm.similarity_search("테스트", k=1)
        print(f"   ✅ 유사도 검색 성공 (결과 수: {len(results)})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 벡터 스토어 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_rag_workflow():
    """RAG 워크플로우 기본 기능 테스트"""
    print("\n🔄 RAG 워크플로우 테스트...")
    
    try:
        from rag_workflow import RAGWorkflow
        
        # RAG 워크플로우 생성
        workflow = RAGWorkflow()
        print("   ✅ RAGWorkflow 초기화 성공")
        
        # 간단한 질문 테스트
        test_query = "테스트에 대해 설명해주세요"
        result = workflow.run_rag(test_query)
        print("   ✅ RAG 워크플로우 실행 성공")
        print(f"   질문: {result['query']}")
        print(f"   답변: {result['answer'][:100]}...")
        print(f"   검색된 문서 수: {len(result['retrieved_docs'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RAG 워크플로우 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_enhanced_workflow():
    """향상된 RAG 워크플로우 테스트"""
    print("\n🎯 향상된 RAG 워크플로우 테스트...")
    
    try:
        from rag_workflow import EnhancedRAGWorkflow
        
        # 향상된 워크플로우 생성
        enhanced_workflow = EnhancedRAGWorkflow()
        print("   ✅ EnhancedRAGWorkflow 초기화 성공")
        
        # 직접 답변 질문 테스트
        direct_query = "안녕하세요"
        result = enhanced_workflow.run_rag(direct_query)
        print("   ✅ 직접 답변 질문 처리 성공")
        print(f"   질문: {result['query']}")
        print(f"   분류: {result['classification']}")
        print(f"   답변: {result['answer'][:100]}...")
        
        # 검색 기반 질문 테스트
        retrieval_query = "테스트에 대해 알려주세요"
        result = enhanced_workflow.run_rag(retrieval_query)
        print("   ✅ 검색 기반 질문 처리 성공")
        print(f"   질문: {result['query']}")
        print(f"   분류: {result['classification']}")
        print(f"   답변: {result['answer'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 향상된 RAG 워크플로우 테스트 실패: {e}")
        traceback.print_exc()
        return False

def test_api_imports():
    """FastAPI 관련 모듈들이 제대로 import되는지 테스트"""
    print("\n🌐 API 모듈 테스트...")
    
    try:
        import main
        print("   ✅ main 모듈 import 성공")
        
        # FastAPI 앱 확인
        if hasattr(main, 'app'):
            print("   ✅ FastAPI 앱 확인됨")
        else:
            print("   ❌ FastAPI 앱을 찾을 수 없음")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ API 모듈 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 LangGraph RAG 시스템 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("모듈 Import", test_imports),
        ("설정", test_configuration),
        ("벡터 스토어", test_vector_store),
        ("RAG 워크플로우", test_rag_workflow),
        ("향상된 워크플로우", test_enhanced_workflow),
        ("API 모듈", test_api_imports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except KeyboardInterrupt:
            print(f"\n\n⚠️  테스트가 사용자에 의해 중단되었습니다.")
            sys.exit(1)
        except Exception as e:
            print(f"   ❌ 예상치 못한 오류: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 테스트 완료: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 성공했습니다!")
        print("\n다음 단계:")
        print("1. uvicorn main:app --reload 로 서버 시작")
        print("2. http://localhost:8000/docs 에서 API 문서 확인")
        print("3. python example_usage.py 로 사용 예제 실행")
    else:
        print("❌ 일부 테스트가 실패했습니다.")
        print("오류를 확인하고 필요한 의존성을 설치하세요.")
        print("\n해결 방법:")
        print("1. pip install -r requirements.txt")
        print("2. .env 파일에 OpenAI API 키 설정")
        print("3. 필요한 경우 Python 환경 재설정")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)