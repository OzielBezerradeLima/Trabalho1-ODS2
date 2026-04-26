import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Local onde o banco de dados será salvo
DB_DIRECTORY = "./db/chroma_db"

def get_vector_db(chunks=None):
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
            persist_directory=DB_DIRECTORY
        )
        print(f"✅ Banco criado com {len(chunks)} chunks em: {DB_DIRECTORY}")
        return vector_db
    else:
        # Carrega o banco do disco
        if os.path.exists(DB_DIRECTORY):
            return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
        else:
            print("❌ Erro: Banco de dados não encontrado.")
            return None