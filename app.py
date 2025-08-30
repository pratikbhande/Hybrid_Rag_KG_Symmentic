import os
import neo4j
from neo4j import GraphDatabase
from PyPDF2 import PdfReader
from groq import Groq
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import spacy

# Set the GROQ_API_KEY directly in the code or retrieve from environment variables
os.environ['GROQ_API_KEY'] = 'gsk_...'

# Function to generate a response using Groq API
def generate_llm_response(query, context, api_key):
    try:
        client = Groq(api_key=api_key)

        prompt = (
            f"You are a helpful assistant with access to a retrieval-augmented system. Based on the context below, "
            f"answer the user's query. Ensure the response is accurate, concise, and well-structured.\n\n"
            f"Context: {context}\n\n"
            f"Query: {query}\n\n"
            f"Answer:"
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.2-3b-preview",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Initialize components
NEO4J_URI = "neo4j+s://dd4e4d25.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ycatl_3YX7XS2y9qdn3WduuK02NDQQZxiuxb53jtGjk"
nlp = spacy.load('en_core_web_sm')

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Function to parse PDF into text
def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() for page in reader.pages])
    return text

# Function to create knowledge graph nodes and relationships
def populate_knowledge_graph(driver, text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]

    print("Extracted Entities:", entities)  # Debugging

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")  # Clear existing graph

        for i, entity in enumerate(entities):
            if entity.strip():
                print(f"Adding Entity: {entity.strip()}")  # Debugging
                session.run(
                    "MERGE (e:Entity {id: $id, name: $name})",
                    id=i,
                    name=entity.strip()
                )

        # Create relationships between adjacent entities (naive approach)
        for i in range(len(entities) - 1):
            if entities[i].strip() and entities[i + 1].strip():
                session.run(
                    "MATCH (e1:Entity {name: $name1}), (e2:Entity {name: $name2}) "
                    "MERGE (e1)-[:RELATED_TO]->(e2)",
                    name1=entities[i].strip(),
                    name2=entities[i + 1].strip()
                )

# Function to query the knowledge graph
def query_knowledge_graph(driver, user_query):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Entity)
            WHERE toLower(n.name) CONTAINS toLower($search_query)
               OR levenshteinDistance(toLower(n.name), toLower($search_query)) < 3
            RETURN n.name
            """,
            search_query=user_query
        )
        return [record["n.name"] for record in result]

# Streamlit App
def visualize_knowledge_graph(driver):
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n")
        graph = nx.Graph()
        for record in result:
            node = record["n"]
            graph.add_node(node["name"])

        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
        for record in result:
            graph.add_edge(record["n"]["name"], record["m"]["name"])

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10)
    st.pyplot(plt)

st.title("Graph-Based Retrieval-Augmented Generation (RAG) System")

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Parse and process document
    doc_text = parse_pdf(uploaded_file)

    st.success("Document processed successfully!")

    # Populate Knowledge Graph
    populate_knowledge_graph(driver, doc_text)

    # Query input
    user_query = st.text_input("Enter your query")

    if st.button("Run Query"):
        if user_query:
            # Perform searches
            kg_results = query_knowledge_graph(driver, user_query)

            # Display results
            st.subheader("Knowledge Graph Results")
            st.write(kg_results)

            # Generate LLM Response
            context = "\n".join(kg_results[:5])
            api_key = os.environ['GROQ_API_KEY']
            response = generate_llm_response(user_query, context, api_key)
            st.subheader("LLM Response")
            st.write(response)

    # Visualize Knowledge Graph
    st.subheader("Knowledge Graph Visualization")
    visualize_knowledge_graph(driver)

# Cleanup
def cleanup():
    if driver:
        driver.close()

cleanup()
