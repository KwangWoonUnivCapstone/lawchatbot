from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import os
from glob import glob

files = glob(os.path.join('', '*.xml'))
loader = DirectoryLoader('C:/Users/user/OneDrive/바탕 화면/lawchatbot/chatbot/data/2021', glob="**/*.xml")
# loader2 = DirectoryLoader('C:/Users/user/OneDrive/바탕 화면/lawchatbot/chatbot/data/2021', glob="**/*.xml")
# loader3 = DirectoryLoader('C:/Users/user/OneDrive/바탕 화면/lawchatbot/chatbot/data/2021', glob="**/*.xml")
# loader4 = DirectoryLoader('C:/Users/user/OneDrive/바탕 화면/lawchatbot/chatbot/data/2021', glob="**/*.xml")
data_docs = []
data_docs = loader.load() # Add this line to define the variable "행정판결_docs"

model_name = "jhgan/ko-sroberta-multitask" # (KorNLU 데이터셋에 학습시킨 한국어 임베딩 모델)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# print(data_docs[0])

data_vectorstore = Chroma.from_documents(documents=data_docs, embedding=embedding_model)
# 행정판결_vectorstore = Chroma.from_documents(documents=행정판결_docs, embedding=embedding_model)
data_retriever = data_vectorstore.as_retriever(search_kwargs={"k": 1})
retrieved_docs = data_retriever.invoke(
    "절도 사건에 관한 판례를 알려줘"
)
