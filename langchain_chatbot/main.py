# Using document from the link: https://arxiv.org/pdf/2312.10997
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

system_prompt = (
    """You are a highly reliable assistant specializing in question-answering tasks. Use the provided context to answer the question accurately and concisely. If the answer cannot be determined from the context, respond with 'I don't know.' Limit your response to a maximum of three sentences.
    \n\n {context}"""
)
rag_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

embeddings = OllamaEmbeddings(model="gemma3:12b")
llm = ChatOllama(model="gemma3:12b")

def get_qa_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(texts, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, question_answer_chain=question_answer_chain)
    return rag_chain


def main():
    rag_chain = get_qa_chain("rags.pdf")
    question = "what are different types of RAG?"
    result = rag_chain.invoke({"question": question})
    print(result)

if __name__ == "__main__":
    main()
