from pdf.extractor import extract_data_from_pdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.database import get_vector_db

def processar_documento(pdf_path):
    print(f"Iniciando processamento de: {pdf_path}")
    
    # Extração 
    dados_pdf = extract_data_from_pdf(pdf_path)

    if dados_pdf is None:
        print("Falha ao ler o PDF.")
        return
        
    _, _, texto_bruto = dados_pdf
    
    # Chunking 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(texto_bruto)
    print(f"📄 Texto dividido em {len(chunks)} pedaços.")
    
    # Armazenamento no Banco Vetorial
    get_vector_db(chunks=chunks)

if __name__ == "__main__":
    # Teste rápido com seu arquivo
    processar_documento("doc_teste.pdf")