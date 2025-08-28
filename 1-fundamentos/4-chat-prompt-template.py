from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

system = ("system", "você é um assistente que responde perguntas em um estilo {style}. Sempre responda em português do Brasil")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="divertido", question="Quem foi Alan Turing?")

for msg in messages:
    print(f"{msg.type}: {msg.content}")

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
result = model.invoke(messages)
print(result.content)