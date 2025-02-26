import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPEN_AI_KEY")

if not openai_api_key:
    raise ValueError("Error: OPEN_AI_KEY not found in .env file.")

openai = OpenAI(api_key=openai_api_key)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_collection")


embed_model = OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-3-small")
Settings.embed_model = embed_model

# Load PDF or directory 
def load_data(path):
    documents = []
    unsupported_files = []

    if os.path.isdir(path):
        print(f"📁 Loading PDFs from directory: {path}")
        all_files = os.listdir(path)
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        unsupported_files = [f for f in all_files if not f.lower().endswith('.pdf')]
        if unsupported_files:
            print(f"--> Skipping unsupported files: {', '.join(unsupported_files)}")

        if not pdf_files:
            print(f"--> No PDF files found in {path}.")
            return documents

        for pdf in pdf_files:
            pdf_path = os.path.join(path, pdf)
            try:
                reader = SimpleDirectoryReader(input_files=[pdf_path])
                docs = reader.load_data()
                documents.extend(docs)
                print(f"✅ Loaded PDF: {pdf}")
            except Exception as e:
                print(f"❌ Error loading {pdf}: {e}")

    elif os.path.isfile(path) and path.lower().endswith('.pdf'):
        print(f"📄 Loaded single PDF: {os.path.basename(path)}")
        try:
            reader = SimpleDirectoryReader(input_files=[path])
            documents = reader.load_data()
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
    else:
        print("--> Invalid path. Please provide a valid PDF file or directory.")

    return documents

# Store embeddings using LlamaIndex
def store_embeddings(documents, collection, pdf_name):
    existing_docs = collection.get(include=["metadatas", "documents"])
    existing_files = [meta.get("pdf_name") for meta in existing_docs["metadatas"] if meta]

    if pdf_name in existing_files:
        print(f"⚡ PDF '{pdf_name}' already processed and stored. Skipping reprocessing.")
        return

    index = VectorStoreIndex.from_documents(documents)

    for i, doc in enumerate(documents):
        try:
            embedding = embed_model.get_text_embedding(doc.text)
            collection.add(
                documents=[doc.text],
                embeddings=[embedding],
                ids=[f"{pdf_name}_chunk_{i}"],
                metadatas=[{"pdf_name": pdf_name}]
            )
        except Exception as e:
            print(f"❌ Error embedding chunk {i}: {e}")
    print(f"✅ PDF '{pdf_name}' processed and stored with embeddings.")

# Query ChromaDB
def query_db(query, collection, top_k=3):
    total_docs = len(collection.get()["ids"])
    adjusted_top_k = min(top_k, total_docs)

    if total_docs == 0:
        return "--> No documents in the database. Please add PDFs first."

    query_embedding = embed_model.get_text_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=adjusted_top_k
    )

    contexts = [doc[0] for doc in results['documents'] if doc]
    return "\n".join(contexts) if contexts else None

# Generate answer using OpenAI LLM with conversation history
def generate_answer_with_history(context, query, history):
    if not context:
        return "--> No relevant information found in the PDF. Try rephrasing your query."

    history_text = ""
    for i, turn in enumerate(history):
        history_text += f"User: {turn['query']}\nAssistant: {turn['answer']}\n"

    prompt = (
        "Use the following context and conversation history to answer the user's question.\n\n"
        f"Context:\n{context[:1000]}\n\n"
        f"Conversation History:\n{history_text}\n"
        f"User's New Question: {query}\n\n"
        "Answer:"
    )

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# Main function with memory
def main():
    print("***RAG System Console App with Memory***")
    print("----------------------------------------")

    conversation_history = []

    # Load PDFs
    while True:
        path = input("Enter PDF file path or directory: ").strip()
        documents = load_data(path)

        if documents:
            for doc in documents:
                pdf_name = os.path.basename(doc.metadata.get("file_path", "unknown.pdf"))
                store_embeddings([doc], collection, pdf_name)

            print("✅ Chunks stored in ChromaDB.")
            break
        else:
            print("--> No valid PDFs found. Please try again.")

    # Query loop with history
    while True:
        query = input("\n🔍 Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("👋 Exiting RAG System. Bye!")
            break

        context = query_db(query, collection)

        answer = generate_answer_with_history(context, query, conversation_history)

        conversation_history.append({"query": query, "answer": answer})

        print("\n💡 Answer:", answer)


if __name__ == "__main__":
    main()

