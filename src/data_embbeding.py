from langchain_community.vectorstores import SKLearnVectorStore
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# Create embeddings for documents and store them in a vector store
def create_embedding(doc_splits):
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )
    vectorstore = FAISS.from_documents(doc_splits, embeddings)
    vectorstore.save_local("faiss_index_")
    persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)
    retriever = persisted_vectorstore.as_retriever()
    """
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(k=4)
    """
    return retriever