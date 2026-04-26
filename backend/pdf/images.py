

import fitz  # PyMuPDF
import os

def extract_images(pdf_path, output_base_dir="imagens"):
    """
    Extrai todas as imagens de um PDF e as salva em um diretório específico.
    Só cria o diretório se imagens forem realmente encontradas.
    Assume que pdf_path já foi validado anteriormente.
    """
    
    # 1. Abre o PDF (Sem verificação de existência do arquivo, assumindo que quem chamou já tratou)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Erro ao abrir o PDF: {e}")
        return

    # 2. Prepara o caminho do diretório de saída (mas NÃO o cria ainda)
    filename = os.path.basename(pdf_path)
    filename_no_ext = os.path.splitext(filename)[0]
    output_dir = os.path.join(output_base_dir, filename_no_ext)

    print(f"Iniciando busca por imagens em '{filename}'...")
    
    total_images = 0
    dir_created = False # Flag para controlar a criação do diretório
    
    for page_index, page in enumerate(doc):
        image_list = page.get_images(full=True)
        
        # Verifica se a página tem imagens
        if not image_list:
            continue
            
        # --- LÓGICA DE CRIAÇÃO PREGUIÇOSA (LAZY) ---
        # Só cria o diretório quando a PRIMEIRA imagem é encontrada
        if not dir_created:
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    print(f"  -> Primeira imagem encontrada. Diretório criado: {output_dir}")
                except OSError as e:
                    print(f"Erro ao criar diretório: {e}")
                    return
            dir_created = True
        # ---------------------------

        print(f"  - Página {page_index + 1}: {len(image_list)} imagem(ns) encontrada(s).")

        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_name = f"imagem_p{page_index + 1}_i{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            total_images += 1

    doc.close()
    
    # Resumo final
    print("=" * 40)
    if total_images > 0:
        print(f"EXTRAÇÃO CONCLUÍDA")
        print(f"Total de imagens salvas: {total_images}")
        print(f"Local: {output_dir}")
    else:
        print("Nenhuma imagem encontrada neste PDF.")
        print("Nenhum diretório foi criado.")
    print("=" * 40)

# --- Exemplo de Uso ---
if __name__ == "__main__":
    test_file = "Desafio Python — Processo Seletivo ADA.pdf"
    
    # Verifica a existência apenas para o teste isolado
    if os.path.exists(test_file):
        extract_images(test_file, output_base_dir="imagens")
    else:
        print(f"Arquivo de teste '{test_file}' não encontrado.")