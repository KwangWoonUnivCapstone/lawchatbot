from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import os
from glob import glob

DATA_PATH = "C:/Users/user/OneDrive/바탕 화면/lawchatbot/chatbot/data"

def modify_xml_file(DATA_PATH):
    # 파일을 읽기 모드로 열기
    with open(DATA_PATH, 'r', encoding='UTF-8') as file:
        contents = file.read()

    # 변경할 내용 지정
    old_content = '<?xml version=1.0 encoding=UTF-8 standalone=yes?>'
    new_content = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'

    # 내용 교체
    updated_contents = contents.replace(old_content, new_content)

    # 파일을 다시 쓰기 모드로 열어 수정된 내용 저장
    with open(DATA_PATH, 'w', encoding='UTF-8') as file:
        file.write(updated_contents)
        print("파일이 성공적으로 수정되었습니다.")
loader = DirectoryLoader(DATA_PATH, glob="**/*.xml")
data_docs = []
data_docs = loader.load() # data_docs에 문서들을 로드
print(data_docs[0])
len(data_docs)

model_name = "jhgan/ko-sroberta-multitask" # (KorNLU 데이터셋에 학습시킨 한국어 임베딩 모델)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


data_vectorstore = Chroma.from_documents(documents=data_docs, embedding=embedding_model)
data_retriever = data_vectorstore.as_retriever(search_kwargs={"k": 1})
retrieved_docs = data_retriever.invoke(
    "절도 사건에 관한 판례를 알려줘"
)
