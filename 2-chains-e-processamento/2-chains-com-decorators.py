from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input_dict:dict) -> dict:
    x = input_dict["x"]
    return {"square_result": x * x}

question_template = PromptTemplate(
    input_variables=["name"],
    template="Olá, eu sou {name}! Me conte uma piada usando meu nome! Responda em português do Brasil."
)

question_template2 = PromptTemplate(
    input_variables=["square_result"],
    template="Me fale sobre o número {square_result}. Responda em português do Brasil de uma forma interessante e divertida."
)

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

chain = question_template | model
chain2 = square | question_template2 | model

result = chain2.invoke({"x":13})
print(result.content)