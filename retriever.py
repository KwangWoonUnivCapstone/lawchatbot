from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
import pandas as pd

# loader = DirectoryLoader('legaldoc\TS_1.판결문', glob="**/*.xml")
# docs = loader.load()
# docs[0]

pd.read_csv('C:/Users/user/OneDrive/바탕 화면/chatbot/legaldoc/TS_1.판결문')