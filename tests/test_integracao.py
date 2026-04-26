from backend.rag.database import get_vector_db
from backend.llm.model import load_llm
from backend.llm.chat import generate_rag_answer

def main():
    print("=== INICIANDO TESTE DE INTEGRAÇÃO RAG ===")
    
    # Definir a pergunta do usuário
    pergunta = "Quais são os níveis de carga e suas caracteristicas"
    print(f"\nPERGUNTA: {pergunta}")

    # Recuperação
    print("\nBuscando informações no banco de dados vetorial...")
    db = get_vector_db()
    
    if not db:
        print("Erro: Banco de dados não encontrado. Rode backend/rag/ingestion.py primeiro.")
        return

    # Busca os 5 pedaços mais relevantes para a pergunta
    docs_recuperados = db.similarity_search(pergunta, k=5)
    
    # Junta os pedaços encontrados em uma única string gigante de contexto
    contexto_real = "\n\n".join([doc.page_content for doc in docs_recuperados])
    
    print("Contexto recuperado com sucesso!")

    # Geração
    print("\nCarregando o modelo de linguagem...")
    model, tokenizer, device = load_llm()

    if not model:
        print("Erro ao carregar o modelo LLM.")
        return

    print("\nGerando a resposta baseada EXCLUSIVAMENTE no contexto do PDF...")
    resposta = generate_rag_answer(model, tokenizer, pergunta, contexto_real, device)

    # Resultado final
    print("\n" + "="*50)
    print("RESPOSTA FINAL DA IA:")
    print("="*50)
    print(resposta)
    print("="*50)

if __name__ == "__main__":
    main()