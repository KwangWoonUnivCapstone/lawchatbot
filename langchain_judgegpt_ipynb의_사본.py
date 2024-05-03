# # 판결문 데이터 다운로드 & 업로드하기
# """

# # /019.법률, 규정 (판결서, 약관 등) 텍스트 분석 데이터/01.데이터/1.Training/원천데이터_230510_add/
# !unzip /TS_1.판결문.zip

# """# LangChain 라이브러리 설치"""

# !pip install langchain openai chromadb tiktoken pypdf unstructured sentence-transformers

# """# 2021년도 형사 판결문 데이터 읽어오기"""

from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('/TS_1.판결문/02.형사/2021', glob="**/*.xml")
docs = loader.load()

docs[0]

len(docs)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sroberta-multitask" # (KorNLU 데이터셋에 학습시킨 한국어 임베딩 모델)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain.vectorstores import Chroma

#vectorstore.delete_collection()  # Collection 삭제
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

"""# Vector Stores를 이용해 형사사건에 대한 전체 판례 찾기"""

retriever = vectorstore.as_retriever()

retrieved_docs = retriever.invoke(
    "절도 사건에 관한 판례를 알려줘"
)
print(retrieved_docs[0].page_content)

retrieved_docs = retriever.invoke(
    "마약 사건에 관한 판례를 알려줘"
)
print(retrieved_docs[0].page_content)



"""# LLM과 연동후 형사사건 판례에 대한 요약 결과를 답변으로 받기

##  OpenAI API Key 설정
"""

OPENAI_KEY = "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"

# 긴 context를 다루기 위해 gpt-3.5-turbo-16k를 이용해서 LLM 설정
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=OPENAI_KEY)

from langchain.prompts import PromptTemplate

template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
만약 답을 모르면 모른다고만 말하고 답을 지어내려고 하지 마십시오.
답변을 형성하는데 참고한 문서들을 근거로 들며 답변해주세요.
맥락: {context}
질문: {question}

도움이 되는 답변:"""
rag_prompt_custom = PromptTemplate.from_template(template)

# RAG chain 설정
from langchain.schema.runnable import RunnablePassthrough

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm

rag_chain.invoke("마약 사건에 관한 판례 번호와 해당 판례의 판결선고일을 알려줘")

rag_chain.invoke("마약 사건에 관한 판례 번호와 해당 판례의 판결 내용을 알려줘")

rag_chain.invoke("2021고합25 판례에 선고된 징역이 몇년인지 알려줘")

rag_chain.invoke("마약 사건에 연류됐는데 처벌이 어느 정도인지 알려줘")

rag_chain.invoke("음주운전 2회로 집행유예 기간 중, 무면허 음주운전으로 적발되었습니다. 저는 업무상 차량이 필수적인 직업을 가지고 있어, 면허 취소 이후에도 집행유예 기간동안 타인의 명의로 몰래 차량을 이용해왔습니다. 최근 다시 음주운전을 하게 되었고, 그러던 중 차 안에서 잠이 들어 경찰에 적발되었습니다. 저는 술을 마시고 약 7시간 뒤에 적발되었고, 측정 결과 혈중알코올농도는 0.083이었습니다. 부모님께도 죄송하고, 특히 자력으로는 생활할 수 없는 가족을 돌봐야 하는 상황에서 실형을 피할 수 있는 방법과 면허를 구제받을 수 있는 방법이 있는지 알고 싶습니다. ")

rag_chain.invoke("""방금 사기를 당한 것 같습니다. 그런데, 상대방이 제게 보낸 계좌가
대포 통장이기에 바로돈을 인출할 수 없다고 합니다.
내일 아침에 은행에서 돈을 인출해서 돌려주겠다고 하는데, 믿어도 될까요?
상대방은 신고만 하지 않으면, 돈을 돌려주겠다고 합니다.
신뢰해도 될지, 아니면 당장 신고를 해야 할지 고민됩니다.""")

from langchain.prompts.chat import ChatPromptTemplate

template = """당신은 판례를 요약하는 유용한 조수입니다. \
사용자가 판례를 전달하면 판례의 주요내용을 요약합니다. \
사건 번호, 판결 선고일, 형량, 범죄사실은 꼭 포함하십시오."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
chain = chat_prompt | llm

result = chain.invoke({"text": docs[0]})
print(result.content)

summary_documents =  []
from langchain.docstore.document import Document

max_len = 16000
for idx, doc in enumerate(docs):
    # 길이 조정
    if len(doc.page_content) > max_len:
      doc.page_content = doc.page_content[:max_len]
    result = chain.invoke({"text": doc})
    print(idx, result.content)
    page = Document(page_content=result.content)

    print(summary_documents.append(page))
    print(rag_chain)