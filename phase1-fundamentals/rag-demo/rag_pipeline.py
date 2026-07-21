import os
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM

DOCS_DIR = "docs"

def load_documents(docs_dir):
    """读取目录下所有txt文件"""
    documents = []
    filenames = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(docs_dir, fname)
            with open(path, "r") as f:
                documents.append(f.read().strip())
                filenames.append(fname)
    return documents, filenames

def build_vector_store(documents, filenames, embeddings):
    """把文档转向量，存进Chroma"""
    doc_embeddings = embeddings.embed_documents(documents)
    client = chromadb.Client()
    collection = client.create_collection(name="rag_demo")
    collection.add(
        embeddings=doc_embeddings,
        documents=documents,
        ids=filenames,
    )
    return collection

def retrieve(query, embeddings, collection, n_results=2):
    """检索最相关的n篇文档"""
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results["documents"][0]

def build_prompt(query, retrieved_docs):
    """把检索到的文档 + 问题拼成给LLM的prompt"""
    context = "\n\n".join(retrieved_docs)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def main():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = OllamaLLM(model="llama3:8b")

    documents, filenames = load_documents(DOCS_DIR)
    print(f"加载了 {len(documents)} 篇文档: {filenames}")

    collection = build_vector_store(documents, filenames, embeddings)

    query = "What is the difference between direct and indirect prompt injection?"
    retrieved_docs = retrieve(query, embeddings, collection, n_results=2)

    print(f"\n查询: {query}")
    print("检索到的文档:")
    for doc in retrieved_docs:
        print(f"- {doc[:80]}...")

    prompt = build_prompt(query, retrieved_docs)
    answer = llm.invoke(prompt)

    print(f"\nLLM回答:\n{answer}")

if __name__ == "__main__":
    main()
