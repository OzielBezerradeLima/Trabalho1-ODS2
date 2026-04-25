from pdf.extractor import extract_text_from_pdf # Ajuste o nome da sua função aqui

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.database import get_vector_db

def processar_documento(pdf_path):
    print(f"Iniciando processamento de: {pdf_path}")
    
    # 1. Extração (Usando seu código existente)
    texto_bruto = extract_text_from_pdf(pdf_path)
    
    # 2. Chunking (Divisão em pedaços)
    # chunk_size de 1000 com overlap de 100 é um bom padrão para ODS
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(texto_bruto)
    print(f"📄 Texto dividido em {len(chunks)} pedaços.")
    
    # 3. Armazenamento no Banco Vetorial
    get_vector_db(chunks=chunks)

if __name__ == "__main__":
    # Teste rápido com seu arquivo
    processar_documento("doc_teste.pdf")