from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

@tool("calculadora", return_direct=True)
def calculadora(expressao: str) -> str:
    """Avalia uma expressão matemática simples e retorna o resultado como string."""
    try:
        resultado = eval(expressao)  # cuidado: apenas para exemplo didático
    except Exception as e:
        return f"Erro: {e}"
    return str(resultado)

@tool("buscar_capital")
def buscar_capital(consulta: str) -> str:
    """Retorna a capital de um determinado país se existir nos dados simulados."""
    dados = {
        "Brasil": "Brasília",
        "França": "Paris",
        "Alemanha": "Berlim",
        "Itália": "Roma",
        "Espanha": "Madri",
        "Estados Unidos": "Washington, D.C."
    }
    for pais, capital in dados.items():
        if pais.lower() in consulta.lower():
            return f"A capital do(a) {pais} é {capital}."
    return "Não sei a capital desse país."

# Inicializar o modelo Gemini
modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)

# Configurar as ferramentas disponíveis
ferramentas = [calculadora, buscar_capital]

prompt = PromptTemplate.from_template(
"""
Responda as seguintes perguntas da melhor forma possível. Você tem acesso às ferramentas abaixo.
Use apenas as informações obtidas das ferramentas, mesmo que você saiba a resposta.
Se a informação não estiver disponível nas ferramentas, diga que não sabe.

{tools}

Use exatamente este formato:

Pergunta: a pergunta que você deve responder
Pensamento: você deve sempre pensar sobre o que fazer
Ação: a ação a ser tomada, deve ser uma destas [{tool_names}]
Entrada da Ação: a entrada para a ação
Observação: o resultado da ação

... (este Pensamento/Ação/Entrada da Ação/Observação pode se repetir)
Pensamento: Agora sei a resposta final
Resposta Final: a resposta final para a pergunta original

Regras:
- Se você escolher uma Ação, NÃO inclua a Resposta Final no mesmo passo.
- Após Ação e Entrada da Ação, espere pela Observação.
- Nunca pesquise na internet. Use apenas as ferramentas fornecidas.

Comece!

Pergunta: {input}
Pensamento:{agent_scratchpad}"""
)

# Criar o agente com o modelo e ferramentas
agente = create_react_agent(modelo, ferramentas, prompt)

# Configurar o executor do agente
executor_agente = AgentExecutor.from_agent_and_tools(
    agent=agente, 
    tools=ferramentas, 
    verbose=True, 
    handle_parsing_errors="Formato inválido. Forneça uma Ação com Entrada da Ação, ou apenas uma Resposta Final.",
    max_iterations=3
)

# Testar o agente
print("\nTestando o agente:\n")
print(executor_agente.invoke({"input": "Qual é a capital do Brasil?"}))
print("\n" + "="*50 + "\n")
print(executor_agente.invoke({"input": "Quanto é 15 + 25?"}))