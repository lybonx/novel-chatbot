# novel_ingest.py (FAISS 버전)
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # ✅ Chroma 대신 FAISS 사용

# 설정
NOVELS_DIR = r"C:\Users\sohye\OneDrive\바탕 화면\langchain\poet\novels"
DB_PATH = "./novel_db_faiss" # ✅ 폴더 이름 변경

def ingest_novels():
    if not os.path.exists(NOVELS_DIR):
        print(f"❌ 오류: 경로를 찾을 수 없습니다 -> {NOVELS_DIR}")
        return

    print(f"[1/4] 소설 파일 로딩 중...")
    loader = DirectoryLoader(NOVELS_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    if not documents:
        print("❌ 파일이 없습니다.")
        return

    print("[2/4] 텍스트 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    print("[3/4] 임베딩 모델 로드 중...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"[4/4] 벡터 DB(FAISS) 저장 중...")
    # ✅ FAISS로 저장
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_function)
    vectorstore.save_local(DB_PATH) 
    
    print("🎉 FAISS 변환 완료! 이제 윈도우 오류가 없을 거예요.")

if __name__ == "__main__":
    ingest_novels()