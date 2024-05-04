from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
pdf_filepath = 'C:/Users/user/OneDrive/바탕 화면/lawchatbot/criminal_law.pdf'
#openssopenssl versionloader = UnstructuredPDFLoader(pdf_filepath)
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

print(len(pages))
