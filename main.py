import os
import faiss
import numpy as np
import neo4j
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set the GROQ_API_KEY directly in the code or retrieve from environment variables
os.environ['GROQ_API_KEY'] = 'gsk_...'

def generate_llm_response(query, context, api_key):
    """
    Generate a response using Groq API based on the query and context.

    Parameters:
        query (str): The user query.
        context (str): Retrieved context for the query.
        api_key (str): The API key for accessing the Groq service.

    Returns:
        str: Generated response.
    """
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
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Function to parse PDF into text
def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() for page in reader.pages])
    return text

# Function to create knowledge graph nodes and relationships
def populate_knowledge_graph(driver, text):
    entities = text.split(".")  # Simplistic split; use NLP for better entity extraction
    with driver.session() as session:
        for i, entity in enumerate(entities):
            session.run(
                "CREATE (:Entity {id: $id, name: $name})",
                id=i,
                name=entity.strip()
            )

def query_knowledge_graph(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return [record["name"] for record in result]

# Function to build a FAISS vector store
def build_faiss_index(doc_texts):
    embeddings = model.encode(doc_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Function to perform TF-IDF search
def tfidf_search(query, docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray()
    ranked_docs = np.argsort(-scores, axis=0).flatten()
    return [(docs[idx], scores[idx][0]) for idx in ranked_docs if scores[idx][0] > 0]

# Hybrid query function
def hybrid_query(user_query, index, embeddings, docs, driver):
    # Query Knowledge Graph
    kg_results = query_knowledge_graph(driver, f"MATCH (n:Entity) WHERE n.name CONTAINS '{user_query}' RETURN n.name")

    # FAISS Vector Search
    query_embedding = model.encode([user_query])
    distances, indices = index.search(query_embedding, 5)  # Top 5 results
    faiss_results = [docs[i] for i in indices.flatten()]

    # TF-IDF Search
    tfidf_results = tfidf_search(user_query, docs)

    # Combine results
    combined_results = list(set(kg_results + faiss_results + [result[0] for result in tfidf_results]))
    return combined_results

# Main pipeline
def process_document(file_path):
    # Parse document
    text = parse_pdf(file_path)

    # Populate Knowledge Graph
    populate_knowledge_graph(driver, text)

    # Build FAISS index
    doc_sentences = text.split(".")
    faiss_index, embeddings = build_faiss_index(doc_sentences)

    # Return pipeline components
    return doc_sentences, faiss_index

# Example Usage
if __name__ == "__main__":
    # Step 1: Load a document
    file_path = "Pratik_Bhande_nov_Resume.docx.pdf"  # Replace with the path to the PDF
    doc_sentences, faiss_index = process_document(file_path)

    # Step 2: Query the system
    user_query = "Explain the main topic."
    hybrid_results = hybrid_query(user_query, faiss_index, None, doc_sentences, driver)

    # Step 3: Generate an LLM response
    context = "\n".join(hybrid_results[:5])  # Use top 5 results as context
    api_key = os.environ['GROQ_API_KEY']
    response = generate_llm_response(user_query, context, api_key)

    print("Hybrid Results:", hybrid_results)
    print("LLM Response:", response)

# Cleanup
if driver:
    driver.close()
