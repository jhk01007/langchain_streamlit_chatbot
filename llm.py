from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_retriever():
    # 벡터 데이터베이스
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name, embedding)
    retriever = database.as_retriever(search_kwargs={'k': 2})
    return retriever


def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)  # 모델
    return llm


def get_dictionary_chain():
    # 질문 변경 체인
    llm = get_llm()
    dicionary = ["사람을 나타내는 표현 -> 거주자"]
    query_change_prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우엔 질문만 리턴해주세요.
        사전: {dicionary}

        질문: {{question}}
    """)
    dictionary_chain = query_change_prompt | llm | StrOutputParser()
    return dictionary_chain

def get_tax_chain():
    # 소득세 체인
    llm = get_llm()
    retriever = get_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    tax_chain = (
            {
                "context": retriever,  # query → context
                "question": RunnablePassthrough()  # query → question
            }
            | prompt
            | llm
    )
    return tax_chain

def get_ai_message(user_message):

    # 최종 체인
    final_chain = get_dictionary_chain() | get_tax_chain()

    # 체인  호출
    return final_chain.invoke({"question": user_message}).content