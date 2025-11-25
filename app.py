import streamlit as st
import os

# --- 1. 기본 설정 ---
st.set_page_config(page_title="소설 캐릭터 챗봇", page_icon="📚")
st.title("📚 소설 속 캐릭터와 대화하기")

# --- 2. 라이브러리 임포트 ---
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 3. 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 입력
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # 모델 선택
    # MODEL_NAME = "gpt-3.5-turbo"
    # 파인튜닝된 모델이 있다면 아래 주석을 풀고 모델 ID를 적으세요
    MODEL_NAME = "gpt-3.5-turbo" 
    
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
        
    # 임베딩 모델 로드
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # FAISS DB 로드
    try:
        vectorstore = FAISS.load_local(
            DB_PATH, 
            embedding_function, 
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        return None

retriever = load_db()

# DB 오류 체크
if not retriever:
    st.error("❌ 'novel_db_faiss' 폴더를 찾을 수 없습니다!")
    st.info("터미널에서 'python novel_ingest.py'를 실행하여 소설을 먼저 저장해주세요.")
    st.stop()

# --- 5. 체인 생성 ---
def get_rag_chain():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)

    system_template = f"""
    당신은 소설 속에 등장하는 '{target_char}'입니다.
    현재 당신은 '{user_role}'와 대화하고 있습니다.

    반드시 아래 [소설 내용]을 참고하여 대답하세요.
    소설에 없는 내용은 지어내지 말고, 모르면 모른다고 하세요.
    
    [지침]
    1. 답변은 2~3문장 이내로 간결하게 하세요.
    2. 소설 속 어투를 유지하세요.

    [소설 내용]
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

# --- 7. 채팅 화면 구현 ---

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
        
        # --- ✅ [디버깅 기능] 여기가 중요합니다! ---
        # 챗봇이 답변하기 전에 검색된 내용을 먼저 보여줍니다.
        try:
            retrieved_docs = retriever.invoke(user_input)
            
            with st.expander(f"🔍 '{target_char}'가 읽은 소설 내용 확인하기 (클릭)"):
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**[참고 문단 {i+1}]**")
                        st.info(doc.page_content) # 파란색 박스로 내용 표시
                else:
                    st.warning("⚠️ 검색된 소설 내용이 없습니다. (DB가 비었거나 관련 내용 없음)")
        except Exception as e:
            st.error(f"검색 중 오류 발생: {e}")
        # ----------------------------------------

        # 체인 실행 및 응답 표시
        chain = get_rag_chain()
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        config = {"configurable": {"session_id": "streamlit_session"}}
        
        with st.spinner("답변 생성 중..."):
            try:
                response = chain_with_history.invoke(
                    {"input": user_input}, 
                    config=config
                )
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"오류 발생: {e}")
