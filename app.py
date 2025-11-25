import streamlit as st
import os

# 1. 화면 설정부터 (로딩 문구 표시)
st.set_page_config(page_title="소설 챗봇", page_icon="📚")
st.title("📚 소설 속 캐릭터와 대화하기")

# 로딩 상태 표시
if "db_loaded" not in st.session_state:
    st.info("🚀 시스템을 부팅 중입니다... (FAISS 엔진 가동)")

# 2. 라이브러리 임포트 (Chroma 제거됨!)
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS # ✅ FAISS 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 설정 (사이드바) ---
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # 모델 선택
    MODEL_NAME = "gpt-3.5-turbo"
    # MODEL_NAME = "ft:gpt-3.5-turbo:your-org:xxxx"

    st.subheader("캐릭터 설정")
    target_char = st.text_input("캐릭터 이름", value="셜록 홈즈")
    user_role = st.text_input("당신의 역할", value="독자")
    
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.store = {}
        st.rerun()

# --- 3. 리소스 로드 (FAISS) ---
@st.cache_resource
def load_db():
    DB_PATH = "./novel_db_faiss" # ✅ FAISS DB 경로
    
    if not os.path.exists(DB_PATH):
        return None
        
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # ✅ FAISS 로드
    vectorstore = FAISS.load_local(
        DB_PATH, 
        embedding_function, 
        allow_dangerous_deserialization=True # 로컬 파일 신뢰 설정
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_db()

if not retriever:
    st.error("❌ DB가 없습니다. 'novel_ingest.py'를 먼저 실행하세요.")
    st.stop()
else:
    # 로딩 완료 시 info 메시지 제거를 위해 session_state 사용
    st.session_state.db_loaded = True

# --- 4. 체인 생성 ---
def get_rag_chain():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)

    system_template = f"""
    당신은 소설 속 '{target_char}'입니다. '{user_role}'와 대화 중입니다.
    소설 내용을 바탕으로 성격과 말투를 연기하세요.
    답변은 2~3문장으로 간결하게 하세요.

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

# --- 5. 세션 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# --- 6. 채팅 화면 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("메시지를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("API 키를 입력해주세요.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            chain = get_rag_chain()
            chain_with_history = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            
            config = {"configurable": {"session_id": "streamlit_session"}}
            
            with st.spinner(f"{target_char}에게 텔레파시 보내는 중..."):
                try:
                    response = chain_with_history.invoke(
                        {"input": user_input}, 
                        config=config
                    )
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"오류: {e}")