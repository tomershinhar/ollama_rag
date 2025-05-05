from src.t_chat import RAGApplication
from src.ollama_init import set_ollama
from src.data_embbeding import create_embedding
from src.data_loader import load_docs


# Initialize the RAG application
BASE_DIRECTORY = "/home/tshinhar/repos/AI_projects/t-chat/data"
directories = [BASE_DIRECTORY]
doc_splits = load_docs(directories=directories)
retriever = create_embedding(doc_splits)
rag_chain = set_ollama()
rag_application = RAGApplication(retriever, rag_chain)
# Example usage
question = "what are Okan's intrests?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)