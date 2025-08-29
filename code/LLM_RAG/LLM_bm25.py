
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import os
import re
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
memory = 6

def init_chain(emb_path,vector_path):
    
    embedding = HuggingFaceEmbeddings(model_name=emb_path)

    # Vector store
    vector_store = Chroma(persist_directory=vector_path, embedding_function=embedding)
    
    # 从 Chroma 中提取所有文档的文本内容
    documents = vector_store._collection.get(include=["documents"])["documents"]

    bm25_retriever  = BM25Retriever.from_texts(documents, k=5)  # k=5 表示返回 top-5 结果

    api_key = "sk-xxxx"
    base_url="https://xxxxx"
    llm = ChatOpenAI(
        model='deepseek-r1-distill-llama-70b', 
        openai_api_key=api_key, 
        openai_api_base=base_url,
    )


    embedding = HuggingFaceEmbeddings(model_name=emb_path)


    instruct_system_prompt = (
        "你是生命周期领域富有经验和知识的专家。"
        "使用以下检索到的上下文来回答问题。"
        "{context}"
        "答案最多使用1句话并保持非常简洁，不能换行。"
    )
    # instr = "你是生命周期领域富有经验和知识的专家。根据你所掌握的知识只用1句话回答问题。不要列出几点来回答，不需要换行"
    instruct_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruct_system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create a chain for passing a list of Documents to a model.
    qa_chain = create_stuff_documents_chain(llm, instruct_prompt) #
    rag_chain = create_retrieval_chain(bm25_retriever, qa_chain)

    return rag_chain, bm25_retriever

def user_in(uin, rag_chain):
    try:
        result = rag_chain.invoke({"input": uin})['answer']
        return result
    except Exception as e:
        print(f"Error in user_in: {e}")
        return "An error occurred while processing your request."

def debug_retrieval(query, retriever):
    docs = retriever.invoke(query)
    print(f"检索查询: {query}")
    print(f"检索到文档数: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"文档{i+1} 相似度分数: {doc.metadata.get('score', 'N/A')}")
        print(f"文档{i+1} 内容预览: {doc.page_content[:100]}...")

def main():
    
    question  = []
    with open("./question.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            question.append(line.strip())
    # question = question[:1]
    
    emb_path = "bge-large-zh-v1.5"
    vector_path = "chroma_bge"
    
    root_path = ""

    rag, retriever = init_chain(emb_path,vector_path)
    
    answers = []
    for ques in question:
        # print(ques)
        response = user_in(ques, rag)
        print(response) 
        # debug_retrieval(ques,retriever)
        answers.append(response.strip())

    file_path = os.path.join(root_path,"DS_RAG_bge_bm25.txt")
    csv_path = os.path.join(root_path,"DS_RAG_bge_bm25.csv")
    with open(file_path,"w",encoding="utf-8") as file:
        for ans in answers:
            line = re.sub(r'\s+', '', ans)
            file.write(line+'\n')
    data = {"ans":answers}
    df = pd.DataFrame(data)
    df.to_csv(csv_path,index=False)

if __name__ == "__main__":
    main()


