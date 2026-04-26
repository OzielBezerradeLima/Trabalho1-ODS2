import torch

def generate_rag_answer(model, tokenizer, question, context, device):
    """
    Gera uma resposta para a pergunta do usuário baseada EXCLUSIVAMENTE 
    no contexto fornecido (os chunks recuperados do banco vetorial).
    """
    if not model or not tokenizer:
        return "Modelo não carregado corretamente."

    print("Processando a resposta...")

    # Prompt
    system_prompt = (
        "Você é um assistente especializado e preciso. "
        "Seu objetivo é responder à pergunta do usuário utilizando APENAS as informações "
        "fornecidas no contexto abaixo. "
        "Se a resposta não estiver no contexto, diga exatamente: 'Desculpe, não encontrei "
        "essa informação no documento fornecido.' Não invente dados."
    )
    
    # Montagem da Mensagem do Usuário
    input_text = f"CONTEXTO DO DOCUMENTO:\n{context}\n\nPERGUNTA:\n{question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]

    # Prepara as entradas no formato de chat do Qwen
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    # Configurações de Geração do Rag
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.1,
        top_p=0.9
    )

    # Decodifica a resposta removendo o prompt de entrada
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return answer