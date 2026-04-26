from backend.llm.model import load_llm
from backend.llm.chat import generate_rag_answer

# Carrega o modelo
model, tokenizer, device = load_llm()

contexto_falso = """
O Sistema Solar é formado por oito planetas principais que orbitam a estrela central, o Sol. 
Os quatro planetas rochosos mais próximos do Sol são Mercúrio, Vênus, Terra e Marte. 
Os quatro planetas gigantes que ficam mais afastados são Júpiter, Saturno, Urano e Netuno, sendo compostos principalmente de gases como hidrogênio e hélio. 
Plutão, que durante décadas foi considerado o nono planeta, foi reclassificado em 2006 pela União Astronômica Internacional (UAI) para a categoria de 'planeta anão'.
"""

pergunta = "Quais são os planetas gigantes gasosos e o que aconteceu com Plutão em 2006?"

resposta = generate_rag_answer(model, tokenizer, pergunta, contexto_falso, device)

print("\n--- RESPOSTA DA IA ---")
print(resposta)