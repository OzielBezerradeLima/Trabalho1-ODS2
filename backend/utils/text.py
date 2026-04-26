# File: text.py
# Description: Main function
# Author: Felipe 
# Data: Dec, 25 

import re
from collections import Counter

def analyze_text(full_text):    
    # Converte para minúsculas
    clean_text = full_text.lower()

    # Regex para pegar apenas palavras (incluindo caracteres acentuados do português)
    # Remove pontuação, números isolados e caracteres especiais
    raw_words = re.findall(r'[a-záàâãéèêíïóôõöúçñ]+', clean_text)

    # Total de palavras (antes de remover as stopwords)
    total_words = len(raw_words)

    # Definição de stopwords (palavras comuns ignoradas na contagem de frequência)
    stopwords = {
        'a', 'o', 'e', 'é', 'de', 'do', 'da', 'os', 'as', 'um', 'uma', 
        'para', 'com', 'não', 'que', 'em', 'no', 'na', 'nos', 'nas', 
        'se', 'por', 'como', 'dos', 'das', 'sua', 'seu', 'ao', 'aos',
        'pelo', 'pela', 'mais', 'foi', 'mas', 'ou', 'tem', 'também', 
        'ser', 'está', 'entre', 'até', 'são', 'era', 'pode'
    }

    # Filtra palavras: remove stopwords e palavras muito curtas (opcional, ex: len > 1)
    filtered_words = [word for word in raw_words if word not in stopwords and len(word) > 1]

    # Tamanho do vocabulário (palavras únicas após a limpeza)
    vocab_size = len(set(filtered_words))

    # Top 10 palavras mais comuns
    counter = Counter(filtered_words)
    top_10 = counter.most_common(10)

    return (total_words, vocab_size, top_10)
