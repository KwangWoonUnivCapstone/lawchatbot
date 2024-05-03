# # �ǰṮ ������ �ٿ�ε� & ���ε��ϱ�
# """

# # /019.����, ���� (�ǰἭ, ��� ��) �ؽ�Ʈ �м� ������/01.������/1.Training/��õ������_230510_add/
# !unzip /TS_1.�ǰṮ.zip

# """# LangChain ���̺귯�� ��ġ"""

# !pip install langchain openai chromadb tiktoken pypdf unstructured sentence-transformers

# """# 2021�⵵ ���� �ǰṮ ������ �о����"""

from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('/TS_1.�ǰṮ/02.����/2021', glob="**/*.xml")
docs = loader.load()

docs[0]

len(docs)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sroberta-multitask" # (KorNLU �����ͼ¿� �н���Ų �ѱ��� �Ӻ��� ��)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain.vectorstores import Chroma

#vectorstore.delete_collection()  # Collection ����
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_model)

"""# Vector Stores�� �̿��� �����ǿ� ���� ��ü �Ƿ� ã��"""

retriever = vectorstore.as_retriever()

retrieved_docs = retriever.invoke(
    "���� ��ǿ� ���� �Ƿʸ� �˷���"
)
print(retrieved_docs[0].page_content)

retrieved_docs = retriever.invoke(
    "���� ��ǿ� ���� �Ƿʸ� �˷���"
)
print(retrieved_docs[0].page_content)



"""# LLM�� ������ ������ �Ƿʿ� ���� ��� ����� �亯���� �ޱ�

##  OpenAI API Key ����
"""

OPENAI_KEY = "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"

# �� context�� �ٷ�� ���� gpt-3.5-turbo-16k�� �̿��ؼ� LLM ����
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=OPENAI_KEY)

from langchain.prompts import PromptTemplate

template = """������ ���� �ƶ��� ����Ͽ� ������ ������ ����Ͻʽÿ�.
���� ���� �𸣸� �𸥴ٰ� ���ϰ� ���� ������� ���� ���ʽÿ�.
�亯�� �����ϴµ� ������ �������� �ٰŷ� ��� �亯���ּ���.
�ƶ�: {context}
����: {question}

������ �Ǵ� �亯:"""
rag_prompt_custom = PromptTemplate.from_template(template)

# RAG chain ����
from langchain.schema.runnable import RunnablePassthrough

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm

rag_chain.invoke("���� ��ǿ� ���� �Ƿ� ��ȣ�� �ش� �Ƿ��� �ǰἱ������ �˷���")

rag_chain.invoke("���� ��ǿ� ���� �Ƿ� ��ȣ�� �ش� �Ƿ��� �ǰ� ������ �˷���")

rag_chain.invoke("2021����25 �Ƿʿ� ����� ¡���� ������� �˷���")

rag_chain.invoke("���� ��ǿ� �����ƴµ� ó���� ��� �������� �˷���")

rag_chain.invoke("���ֿ��� 2ȸ�� �������� �Ⱓ ��, ������ ���ֿ������� ���ߵǾ����ϴ�. ���� ������ ������ �ʼ����� ������ ������ �־�, ���� ��� ���Ŀ��� �������� �Ⱓ���� Ÿ���� ���Ƿ� ���� ������ �̿��ؿԽ��ϴ�. �ֱ� �ٽ� ���ֿ����� �ϰ� �Ǿ���, �׷��� �� �� �ȿ��� ���� ��� ������ ���ߵǾ����ϴ�. ���� ���� ���ð� �� 7�ð� �ڿ� ���ߵǾ���, ���� ��� ���߾��ڿó󵵴� 0.083�̾����ϴ�. �θ�Բ��� �˼��ϰ�, Ư�� �ڷ����δ� ��Ȱ�� �� ���� ������ ������ �ϴ� ��Ȳ���� ������ ���� �� �ִ� ����� ���㸦 �������� �� �ִ� ����� �ִ��� �˰� �ͽ��ϴ�. ")

rag_chain.invoke("""��� ��⸦ ���� �� �����ϴ�. �׷���, ������ ���� ���� ���°�
���� �����̱⿡ �ٷε��� ������ �� ���ٰ� �մϴ�.
���� ��ħ�� ���࿡�� ���� �����ؼ� �����ְڴٰ� �ϴµ�, �Ͼ �ɱ��?
������ �Ű� ���� ������, ���� �����ְڴٰ� �մϴ�.
�ŷ��ص� ����, �ƴϸ� ���� �Ű� �ؾ� ���� ��ε˴ϴ�.""")

from langchain.prompts.chat import ChatPromptTemplate

template = """����� �Ƿʸ� ����ϴ� ������ �����Դϴ�. \
����ڰ� �Ƿʸ� �����ϸ� �Ƿ��� �ֿ䳻���� ����մϴ�. \
��� ��ȣ, �ǰ� ������, ����, ���˻���� �� �����Ͻʽÿ�."""
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
    # ���� ����
    if len(doc.page_content) > max_len:
      doc.page_content = doc.page_content[:max_len]
    result = chain.invoke({"text": doc})
    print(idx, result.content)
    page = Document(page_content=result.content)

    print(summary_documents.append(page))
    print(rag_chain)