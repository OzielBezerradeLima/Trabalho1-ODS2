import fitz  # PyMuPDF
import os

def extract_data_from_pdf(file_path):
    """
    Abre o PDF e retorna os dados brutos:
    (tamanho_bytes, numero_paginas, texto_completo)
    """
    print(f"DEBUG: Tentando abrir o arquivo em: {file_path}")

    # 1. Tamanho do arquivo
    try:
        file_size_bytes = os.path.getsize(file_path)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: O arquivo '{file_path}' não foi encontrado pelo sistema (os.path).")
        return None

    # 2. Abrir PDF
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"ERRO CRÍTICO: Falha ao abrir o PDF com PyMuPDF. Detalhe: {e}")
        return None

    # 3. Contar páginas
    try:
        num_pages = doc.page_count
        print(f"DEBUG: PDF aberto com sucesso. Páginas encontradas: {num_pages}")
    except Exception as e:
        print(f"ERRO: Não foi possível ler o número de páginas. {e}")
        doc.close()
        return None
    
    # 4. Extrair texto
    full_text = ""
    try:
        for page in doc:
            full_text += page.get_text() + " "
    except Exception as e:
        print(f"ERRO: Falha ao extrair texto das páginas. {e}")
        doc.close()
        return None
    
    doc.close()

    # Verifica se extraiu algo
    if not full_text.strip():
        print("AVISO: O PDF foi lido, mas nenhum texto foi encontrado (pode ser imagem escaneada).")

    # RETORNA os dados para quem chamou (o main)
    # É essencial retornar exatamente 3 valores
    return file_size_bytes, num_pages, full_text