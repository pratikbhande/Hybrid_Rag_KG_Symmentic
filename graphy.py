import os
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set environment variables
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["NEO4J_URI"] = "neo4j+s://..."
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "..."

# Initialize Neo4j graph and embeddings
st.title("PDF Question Answering with Neo4j and LangChain")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

graph = Neo4jGraph()
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if uploaded_file:
    st.sidebar.success("PDF Uploaded Successfully")

    # Load PDF documents using PyPDF2
    st.write("Loading and splitting PDF...")
    pdf_reader = PdfReader(uploaded_file)  # Use PyPDF2 to read the uploaded file
    pdf_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_text(pdf_text)

    st.write("Processing documents into graph...")

    # Initialize LLM for graph transformation
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Add documents to the graph
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    # Set up the vector index
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    st.write("Graph and Vector Index are ready.")

    # Define retriever functions
    def generate_full_text_query(input: str) -> str:
        words = [word for word in input.split() if word]
        return " AND ".join([f"{word}~2" for word in words])

    def structured_retriever(question: str) -> str:
        entities = []  # Mock entity extraction as we skipped this part for brevity
        result = ""
        for entity in entities:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50""",
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def retriever(question: str) -> str:
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        return f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}"""

    # Build the chain
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        """Given the following conversation and a follow-up question, rephrase the follow-up question to be standalone.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    )

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            RunnablePassthrough.assign(chat_history=lambda x: [
                HumanMessage(content=msg[0]) if i % 2 == 0 else AIMessage(content=msg[1])
                for i, msg in enumerate(x.get("chat_history", []))
            ]) | CONDENSE_QUESTION_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel({
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }) | prompt | llm | StrOutputParser()
    )

    # Input form for questions
    st.write("Ask a question about the uploaded PDF:")
    question = st.text_input("Question")

    if question:
        st.write("Processing your query...")
        response = chain.invoke({"question": question})
        st.write("Answer:", response)
