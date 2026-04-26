import os
import sys

# --- IMPORTAÇÕES ---
# Certifique-se de que seus arquivos auxiliares (arguments.py, files.py)
# estão com os nomes de função corretos (get_args, validate_path)
from cli.arguments import get_args 
from utils.files import validate_path
from pdf.extractor import extract_data_from_pdf
from utils.text import analyze_text
from pdf.images import extract_images

# Importações de IA
from llm.model import load_llm
from llm.chat import generate_rag_answer

def print_header():
    print("\n" + "*" * 40)
    print("      PROJETO ADA - ANALISADOR DE PDF")
    print("*" * 40 + "\n")

def display_results(filename, file_size, num_pages, total_words, vocab_size, top_10):
    """
    Exibe os resultados estatísticos básicos.
    """
    print("=" * 40)
    print(f"RELATÓRIO DE ANÁLISE: {filename}")
    print("=" * 40)
    print(f"1. Tamanho do arquivo:   {file_size:,} bytes".replace(",", "."))
    print(f"2. Total de páginas:     {num_pages}")
    print(f"3. Total de palavras:    {total_words}")
    print(f"4. Vocabulário (únicas): {vocab_size}")
    print("-" * 40)
    print("5. Top 10 Palavras mais frequentes (sem stopwords):")
    for i, (word, freq) in enumerate(top_10, 1):
        print(f"   {i}. {word}: {freq} vezes")
    print("=" * 40)

def main():
    # 1. Cabeçalho e Argumentos
    print_header()
    
    try:
        args = get_args() 
        pdf_path = args.pdf_path 
    except AttributeError:
        print("ERRO CRÍTICO: Seu 'arguments.py' está desatualizado. Renomeie 'caminho_pdf' para 'pdf_path' dentro dele.")
        return

    # 2. Validação do arquivo
    if not validate_path(pdf_path):
        sys.exit(1)

    # 3. Extração de Imagens
    try:
        extract_images(pdf_path)
    except Exception as e:
        print(f"Aviso: A extração de imagens falhou ({e}), continuando para a análise de texto...")

    # 4. Extração de Dados do PDF
    print("Extraindo texto do PDF...")
    
    # Chamamos o extrator passando o caminho explicitamente
    data_pdf = extract_data_from_pdf(pdf_path)
    
    # Se a extração retornar None, paramos AQUI.
    if data_pdf is None:
        print("Falha ao extrair dados do PDF. Verifique se o arquivo é um PDF válido.")
        return

    # Desempacota os dados apenas se houver sucesso
    file_size, num_pages, full_text = data_pdf

    # 5. Análise de Texto
    total_words, vocab_size, top_10 = analyze_text(full_text)

    # 6. Exibição de Dados Básicos
    display_results(
        filename=os.path.basename(pdf_path),
        file_size=file_size,
        num_pages=num_pages,
        total_words=total_words,
        vocab_size=vocab_size,
        top_10=top_10
    )

    # --- PARTE DA IA ---

    # 7. Carregamento do Modelo
    print("\n--- INICIANDO RESUMO COM IA ---")
    print("(Nota: A primeira execução fará o download do modelo, pode levar alguns minutos)")
    
    model, tokenizer, device = load_llm()
    
    if model:
        # 8. Geração de Resumo
        summary = generate_summary(model, tokenizer, full_text, device)
        
        print("\n" + "="*40)
        print("RESUMO GERADO PELA IA:")
        print("="*40)
        print(summary)
        print("="*40)

        # 9. Salvar em arquivo (usando a flag --save)
        if args.save:
            output_name = f"resumo_{os.path.basename(pdf_path)}.txt"
            try:
                with open(output_name, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"\nResumo salvo em: {output_name}")
            except Exception as e:
                print(f"Erro ao salvar arquivo: {e}")
    else:
        print("Não foi possível carregar o modelo de IA. Resumo ignorado.")

if __name__ == "__main__":
    main()