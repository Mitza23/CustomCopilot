import os

from langchain_core.globals import set_verbose, set_debug
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import chromadb

set_debug(True)
set_verbose(True)


class RAGSystem:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "qwen2.5-coder:7b", chunk_size: int = 1024, chunk_overlap: int = 100):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful Java coding assistant that answers requests based on the information in the document given by the user.",
                ),
                (
                    "human",
                    "Here is the document pieces you need to take as guidelines: {context}. Follow them closely. The user is working in Java.\nRequest: {question}",
                ),
            ]
        )

        self.vector_store = Chroma(
            embedding_function=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )

    def ingest(self, file_path: str, file_name: str):
        extension = os.path.splitext(file_name)[-1].lower()
        if extension == ".pdf":
            docs = PyPDFLoader(file_path=file_path).load()
        elif extension == ".txt":
            docs = TextLoader(file_path=file_path).load()
        else:
            raise ValueError(f"Unsupported file: {file_name}")

        chunks = self.text_splitter.split_documents(docs)
        for chunk in chunks:
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["source"] = file_name

        chunks = filter_complex_metadata(chunks)
        self.vector_store.add_documents(documents=chunks)

    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding_function=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={"k": 10}
            # , "score_threshold": 0.5},
        )

        # # Get chunks with similarity scores using similarity_search_with_score
        # retrieved_chunks_with_scores = self.vector_store.similarity_search_with_score(
        #     query, k=10
        # )
        #
        # print("\nRetrieved chunks with similarity scores:")
        # for i, (chunk, score) in enumerate(retrieved_chunks_with_scores, 1):
        #     print(f"\nChunk {i}:")
        #     print(f"Similarity Score: {score:.4f}")
        #     print(f"Content:\n{chunk.page_content}")
        #     print(f"Metadata: {chunk.metadata}")
        #     print("-" * 50)

        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
        )
        return self.chain.invoke(query)
