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

set_debug(False)
set_verbose(False)


class RAGSystem:
    vector_store = None
    retriever = None
    chain = None
    top_k = None

    def __init__(self, llm_model: str = "qwen2.5-coder:7b",
                 chunk_size: int = 512,
                 chunk_overlap: int = 60,
                 top_k: int = 5):
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

        self.top_k = top_k
        self.retriever = self.vector_store.as_retriever(
            # search_type="similarity_score_threshold",
            search_kwargs={"k": top_k}
            # , "score_threshold": 0.5},
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

    def ask(self, query: str, include_chunks: bool = False):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding_function=FastEmbedEmbeddings()
            )

        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
        )
        output = self.chain.invoke(query)

        if include_chunks:
            retrieved_chunks_with_scores = self.vector_store.similarity_search(
                query, k=self.top_k
            )
            return output, retrieved_chunks_with_scores
        else:
            return output
        #
        # Get chunks with similarity scores using similarity_search_with_score
        # retrieved_chunks_with_scores = self.vector_store.similarity_search(
        #     query, k=self.top_k
        # )
        # print("\nRetrieved chunks with similarity scores:")
        # for i, (chunk, score) in enumerate(retrieved_chunks_with_scores, 1):
        #     print(f"\nChunk {i}:")
        #     print(f"Similarity Score: {score:.4f}")
        #     print(f"Content:\n{chunk.page_content}")
        #     print(f"Metadata: {chunk.metadata}")
        #     print("-" * 50)