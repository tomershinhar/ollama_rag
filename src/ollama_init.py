from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Define the prompt template for the LLM
def set_ollama():
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question, but don't metion the document in your answer.
        If the question is unrelated to the information in the documents, just say so.
        Keep the answers to the point, only talk about what you do know and not about what you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    # Initialize the LLM with Llama 3.1 model
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )
    # Create a chain combining the prompt template and LLM
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain