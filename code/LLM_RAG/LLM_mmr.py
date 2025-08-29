
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import re
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda"
memory = 6

def init_chain(emb_path,vector_path,):
    try:
        # 加载环境变量并初始化模型

        
        api_key = "sk-xxxx"
        base_url="https://xxxxx"
        llm = ChatOpenAI(
            model='deepseek-r1-distill-llama-70b',  # model_name
            openai_api_key=api_key, 
            openai_api_base=base_url,
        )


        embedding = HuggingFaceEmbeddings(model_name=emb_path)
        retriever = Chroma(persist_directory=vector_path, embedding_function=embedding)

        retriever = retriever.as_retriever(search_type="mmr", search_kwargs={"k": 5})

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
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        return rag_chain
    
    except Exception as e:
        print(f"Error in init_chain: {e}")
        return None

def user_in(uin, rag_chain):
    try:
        result = rag_chain.invoke({"input": uin})['answer']
        return result
    except Exception as e:
        print(f"Error in user_in: {e}")
        return "An error occurred while processing your request."

def main():
    
    question  = []
    with open("question.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            question.append(line.strip())
    # question = question[:1]
    
    emb_path = "/bge-large-zh-v1.5"
    vector_path = "/chroma_bge"
    
    root_path = "///"

    rag = init_chain(emb_path,vector_path)
    
    answers = []
    for ques in question:
        # print(ques)
        response = user_in(ques, rag)
        print(response) 
        answers.append(response.strip())

    file_path = os.path.join(root_path,"DS_RAG_bge_mmr.txt")
    csv_path = os.path.join(root_path,"DS_RAG_bge_mmr.csv")
    with open(file_path,"w",encoding="utf-8") as file:
        for ans in answers:
            line = re.sub(r'\s+', '', ans)
            file.write(line+'\n')
    data = {"ans":answers}
    df = pd.DataFrame(data)
    df.to_csv(csv_path,index=False)

if __name__ == "__main__":
    main()

