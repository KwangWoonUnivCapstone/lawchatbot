from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
pdf_filepath = '/Users/eungwanhwi/Desktop/lawchatbotproject/venv/lawchatbot/criminal_law.pdf'
#openssopenssl versionloader = UnstructuredPDFLoader(pdf_filepath)
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

len(pages)
