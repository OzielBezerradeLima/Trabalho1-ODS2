# File: main.py
# Description: Command Line Interface functions
# Author: Felipe 
# Data: Dec, 25

import argparse

def get_args():
    """
    LEGADO: mantido apenas para compatibilidade.
    Analisa os argumentos da linha de comando.
    Retorna:
        args: Os argumentos analisados com os atributos .pdf_path e .save.
    """
    parser = argparse.ArgumentParser(description="Ferramenta de Análise de PDF ADA")
    
    # Argumento obrigatório: o arquivo PDF
    # Nome mantido como 'pdf_path' para padrão interno do código
    parser.add_argument("pdf_path", help="Caminho para o arquivo PDF a ser analisado")
    
    # Argumento opcional: salvar output
    # Flag mantida como '--save'
    parser.add_argument("--save", action="store_true", help="Salva o resumo gerado em um arquivo .txt")

    return parser.parse_args()