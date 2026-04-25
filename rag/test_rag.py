from rag.database import get_vector_db

# 1. Carrega o banco que você já criou
db = get_vector_db()

# 2. Faz uma pergunta de teste sobre a ODS 2
query = "Quais são as principais metas para combater a fome?"
docs = db.similarity_search(query, k=3) # 'k=3' traz os 3 trechos mais relevantes

# 3. Exibe o que ele encontrou
for i, doc in enumerate(docs):
    print(f"--- Trecho {i+1} ---")
    print(doc.page_content)