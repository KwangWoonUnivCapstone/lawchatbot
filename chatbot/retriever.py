from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from glob import glob

files = glob(os.path.join('./', '*.xml'))
loader = DirectoryLoader('./data', glob="**/*.xml")

형사판결_docs = loader.load()
행정판결_docs = []  # Add this line to define the variable "행정판결_docs"

model_name = "jhgan/ko-sroberta-multitask" # (KorNLU 데이터셋에 학습시킨 한국어 임베딩 모델)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
행정판결_docs[0]

형사판결_vectorstore = Chroma.from_documents(documents=형사판결_docs, embedding=embedding_model)
# 행정판결_vectorstore = Chroma.from_documents(documents=행정판결_docs, embedding=embedding_model)
