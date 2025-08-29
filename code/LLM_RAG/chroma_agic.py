'''
生成数据库
'''
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import sys
os.chdir(sys.path[0])

# Load the enivronment variables and API keys 
load_dotenv(".env") 
key = 1

# Call the model and parser for extracting the output of LLM
parser = StrOutputParser()

# Store the vectrostores locally
def chroma_save(path,key=key):
    #Load the docs
    loader = PyPDFLoader(path)
    doc = loader.load()

    # Split the doc content
    tex_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splits = tex_splitter.split_documents(doc)

    # Store the content
    embedding = HuggingFaceEmbeddings(model_name = "/home/zhangxj/models/acge_text_embedding")# 更换embedding 
    vs = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="./chromaDB")
    vs.add_documents(documents=splits)
    vs.as_retriever()
    print("saved: " + str(path))


def main():
    data = "./LCAdata" #os.path.join('resources', 'pdfs')
    docs = os.listdir(data)

    # save every file to the loacal database
    for doc in docs:
        doc_path = os.path.join(data,doc)
        chroma_save(doc_path)



if __name__ == "__main__":
    main()
