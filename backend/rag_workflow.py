from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

from config import settings
from vector_store import vector_store_manager

# State 정의
class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    retrieved_docs: List[Document]
    context: str
    answer: str
    classification: str

class RAGWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.temperature
        )
        
        # RAG 프롬프트 템플릿
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 도움이 되는 AI 어시스턴트입니다. 
            주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공하세요.
            
            컨텍스트:
            {context}
            
            답변 시 다음 규칙을 따르세요:
            1. 컨텍스트에 있는 정보만을 사용하여 답변하세요
            2. 컨텍스트에서 찾을 수 없는 정보라면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 말하세요
            3. 답변은 명확하고 구체적으로 작성하세요
            4. 한국어로 답변하세요"""),
            ("human", "{question}")
        ])
        
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우를 생성합니다."""
        workflow = StateGraph(RAGState)
        
        # 노드 추가
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)
        
        # 엣지 설정
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """문서 검색 단계"""
        query = state["query"]
        
        # 벡터 스토어에서 관련 문서 검색
        retrieved_docs = vector_store_manager.similarity_search(query, k=4)
        
        # 검색된 문서들을 컨텍스트로 변환
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "context": context
        }
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """답변 생성 단계"""
        query = state["query"]
        context = state["context"]
        
        # RAG 체인 실행
        chain = self.rag_prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": query
        })
        
        # 메시지 형태로 답변 추가
        ai_message = AIMessage(content=answer)
        
        return {
            **state,
            "answer": answer,
            "messages": state["messages"] + [ai_message]
        }
    
    def run_rag(self, query: str) -> dict:
        """RAG 워크플로우를 실행합니다."""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "retrieved_docs": [],
            "context": "",
            "answer": ""
        }
        
        result = self.workflow.invoke(initial_state)
        
        return {
            "query": result["query"],
            "answer": result["answer"],
            "retrieved_docs": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result["retrieved_docs"]
            ],
            "context": result["context"]
        }

# Enhanced RAG Workflow with Routing
class EnhancedRAGWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.temperature
        )
        
        # 질문 분류 프롬프트
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """사용자의 질문을 분석하여 다음 중 하나로 분류하세요:
            
            1. RETRIEVAL: 문서나 지식에서 정보를 찾아야 하는 질문
            2. DIRECT: 일반적인 대화나 간단한 질문으로 검색이 필요 없는 질문
            
            분류 결과만 답하세요 (RETRIEVAL 또는 DIRECT)."""),
            ("human", "{question}")
        ])
        
        self.direct_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친근하게 답변하세요."),
            ("human", "{question}")
        ])
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 도움이 되는 AI 어시스턴트입니다. 
            주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공하세요.
            
            컨텍스트:
            {context}
            
            답변 시 다음 규칙을 따르세요:
            1. 컨텍스트에 있는 정보만을 사용하여 답변하세요
            2. 컨텍스트에서 찾을 수 없는 정보라면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 말하세요
            3. 답변은 명확하고 구체적으로 작성하세요
            4. 한국어로 답변하세요"""),
            ("human", "{question}")
        ])
        
        self.workflow = self._create_enhanced_workflow()
    
    def _create_enhanced_workflow(self) -> StateGraph:
        """향상된 RAG 워크플로우를 생성합니다."""
        workflow = StateGraph(RAGState)
        
        # 노드 추가
        workflow.add_node("classify", self._classify_question)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate_rag", self._generate_rag_answer)
        workflow.add_node("generate_direct", self._generate_direct_answer)
        
        # 엣지 설정
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_question,
            {
                "retrieval": "retrieve",
                "direct": "generate_direct"
            }
        )
        workflow.add_edge("retrieve", "generate_rag")
        workflow.add_edge("generate_rag", END)
        workflow.add_edge("generate_direct", END)
        
        return workflow.compile()
    
    def _classify_question(self, state: RAGState) -> RAGState:
        """질문을 분류합니다."""
        query = state["query"]
        
        chain = self.classifier_prompt | self.llm | StrOutputParser()
        classification = chain.invoke({"question": query}).strip().upper()
        
        return {
            **state,
            "classification": classification
        }
    
    def _route_question(self, state: RAGState) -> str:
        """질문 분류 결과에 따라 라우팅합니다."""
        classification = state.get("classification", "RETRIEVAL")
        
        if classification == "DIRECT":
            return "direct"
        else:
            return "retrieval"
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """문서 검색 단계"""
        query = state["query"]
        
        retrieved_docs = vector_store_manager.similarity_search(query, k=4)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "context": context
        }
    
    def _generate_rag_answer(self, state: RAGState) -> RAGState:
        """RAG 기반 답변 생성"""
        query = state["query"]
        context = state["context"]
        
        chain = self.rag_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": query
        })
        
        ai_message = AIMessage(content=answer)
        
        return {
            **state,
            "answer": answer,
            "messages": state["messages"] + [ai_message]
        }
    
    def _generate_direct_answer(self, state: RAGState) -> RAGState:
        """직접 답변 생성"""
        query = state["query"]
        
        chain = self.direct_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": query})
        
        ai_message = AIMessage(content=answer)
        
        return {
            **state,
            "answer": answer,
            "messages": state["messages"] + [ai_message]
        }
    
    def run_rag(self, query: str) -> dict:
        """향상된 RAG 워크플로우를 실행합니다."""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "classification": ""
        }
        
        result = self.workflow.invoke(initial_state)
        
        return {
            "query": result["query"],
            "answer": result["answer"],
            "classification": result.get("classification", ""),
            "retrieved_docs": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result.get("retrieved_docs", [])
            ],
            "context": result.get("context", "")
        }

# 워크플로우 인스턴스
rag_workflow = RAGWorkflow()
enhanced_rag_workflow = EnhancedRAGWorkflow()