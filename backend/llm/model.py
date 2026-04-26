import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm():
    """
    Carrega o modelo LLM local e o tokenizador do Hugging Face.
    Retorna:
        model, tokenizer, device
    """
    # ID do modelo no Hugging Face. 
    # O Qwen2.5-1.5B-Instruct é excelente para português e leve (roda na maioria das CPUs/GPUs).
    # Você pode mudar para "TinyLlama/TinyLlama-1.1B-Chat-v1.0" se precisar de algo ainda menor.
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Carregando modelo '{model_id}'... Isso pode demorar um pouco.")

    try:
        # Detecta se há GPU disponível (CUDA para Nvidia, MPS para Mac), caso contrário usa CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"Usando dispositivo: {device.upper()}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Carrega o modelo com mapeamento automático de dispositivo
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True
        )

        return model, tokenizer, device

    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None, None, None