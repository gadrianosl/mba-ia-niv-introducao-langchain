from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],
    template="Olá, eu sou {name}! Me conte uma piada usando meu nome! Responda em português do Brasil."
)

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

chain = question_template | model

result = chain.invoke({"name": "Adriano"})
print(result.content)