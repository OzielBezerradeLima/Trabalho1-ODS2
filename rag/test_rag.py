from rag.database import get_vector_db

# Carrega o banco
db = get_vector_db()

# Pergunta de teste
query = "Quais são os níveis de carga e suas caracteristicas"
docs = db.similarity_search(query, k=5)

# Exibir Resultado
for i, doc in enumerate(docs):
    print(f"--- Trecho {i+1} ---")
    print(doc.page_content)