import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Local onde o banco de dados será salvo
DB_DIRECTORY = "./db/chroma_db"

def get_vector_db(chunks=None, persist_directory=DB_DIRECTORY, collection_name="default"):
    """
    Se 'chunks' for passado, cria um novo banco. 
    Se for None, apenas carrega o banco existente.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if chunks:
        # Cria e persiste o banco
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"✅ Banco criado com {len(chunks)} chunks em: {persist_directory}")
        return vector_db
    else:
        # Carrega o banco do disco
        if os.path.exists(persist_directory):
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
        else:
            print("❌ Erro: Banco de dados não encontrado.")
            return None


def retrieve_context(vector_db, query, k=5):
    """Recupera contexto relevante no banco vetorial para uma pergunta."""
    docs = vector_db.similarity_search(query, k=k)
    return docs, "\n\n".join([doc.page_content for doc in docs])