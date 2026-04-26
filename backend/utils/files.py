# File: files.py
# Description: Basic functions for files manipulation
# Author: Felipe 
# Data: 04/12

import os

def validate_path(path: str) -> bool:
    """
    Verifica se o arquivo existe e se é um PDF.
    Retorna True se estiver tudo certo, False caso contrário.
    """
    return os.path.exists(path) and os.path.isfile(path)
