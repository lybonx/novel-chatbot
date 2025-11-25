import streamlit as st
import os

# --- 1. 기본 설정 및 화면 구성 ---
st.set_page_config(page_title="소설 캐릭터 챗봇", page_icon="📚")
st.title("📚 소설 속 캐릭터와 대화하기")

# 로딩 상태 표시를 위한 공간
status_container = st.empty()

# --- 2. 라이브러리 임포트 (무거운 작업) ---
if "imports_done" not in st.session_state:
    status_container.info("🚀 시스템 초기화 중... (AI 모델 로딩)")

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

st.session_state.imports_done = True
status_container.empty() # 로딩 문구 삭제


# --- 3. 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 입력
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # 모델 선택
    model_name = "gpt-3.5-turbo"
    # model_name = "ft:gpt-3.5-turbo:your-org:xxxx" # 파인튜닝 모델이 있다면 주석 해제
    
    st.divider()
    
    st.subheader("🎭 캐릭터 설정")
    target_char = st.text_input("캐릭터 이름", value="셜록 홈즈")
    user_role = st.text_input("당신의 역할", value="독자")
    
    st.divider()
    
    if st.button("🗑️ 대화 내용 초기화"):
        st.session_state.messages = []
        st.session_state.store = {}
        st.rerun()


# --- 4. 데이터베이스(FAISS) 로드 ---
@st.cache_resource
def load_db():
    DB_PATH = "./novel_db_faiss"
    
    if not os.path.exists(DB_PATH):
        return None
        
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    try:
        vectorstore = FAISS.load_local(
            DB_PATH, 
            embedding_function, 
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3}) # 관련 내용 3개 검색
    except Exception as e:
        return None

retriever = load_db()

# DB가 없을 경우 경고
if not retriever:
    st.error("❌ 데이터베이스를 찾을 수 없습니다!")
    st.warning("👉 프로젝트 폴더에 'novel_db_faiss' 폴더가 있는지 확인하세요.")
    st.info("💡 해결법: 터미널에서 'python novel_ingest.py'를 실행하여 소설을 먼저 저장해야 합니다.")
    st.stop()


# --- 5. 체인 생성 함수 ---
def get_rag_chain():
    llm = ChatOpenAI(model=model_name, temperature=0.7)

    # 소설 내용을 강제로 참고하도록 프롬프트 강화
    system_template = f"""
    당신은 소설 속에 등장하는 '{target_char}'입니다.
    현재 당신은 '{user_role}'와 대화하고 있습니다.

    아래 [참고한 소설 내용]을 바탕으로 대답하세요.
    소설에 없는 내용은 지어내지 말고, 캐릭터의 말투와 성격을 유지하세요.

    [지침]
    1. 답변은 2~3문장으로 간결하게 하세요.
    2. 소설 속 상황을 자연스럽게 언급하세요.

    [참고한 소설 내용]
    {{context}}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("input") | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# --- 6. 세션 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


# --- 7. 채팅 UI ---
# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요..."):
    # 1. 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ 왼쪽 사이드바에 OpenAI API 키를 입력해주세요.")
        st.stop()

    # 3. AI 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # --- [디버깅 기능] RAG 검색 결과 미리보기 ---
        # 챗봇이 대답하기 전에 무엇을 읽었는지 확인
        try:
            retrieved_docs = retriever.invoke(user_input)
            with st.expander(f"🔍 '{target_char}'가 읽은 소설 내용 확인하기 (클릭)"):
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**[참고 {i+1}]**")
                        st.caption(doc.page_content[:300] + "...") # 너무 길면 자름
                else:
                    st.warning("⚠️ 관련된 소설 내용을 찾지 못했습니다.")
        except Exception as e:
            st.error(f"검색 중 오류: {e}")
        # ---------------------------------------------

        # 체인 실행
        chain = get_rag_chain()
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        config = {"configurable": {"session_id": "streamlit_session"}}
        
        with st.spinner(f"{target_char}(이)가 생각 중..."):
            try:
                response = chain_with_history.invoke(
                    {"input": user_input}, 
                    config=config
                )
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"답변 생성 오류: {e}")
