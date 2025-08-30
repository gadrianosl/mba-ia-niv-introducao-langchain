from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ```{initial_text}````"
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n ```{text}```"
)

llm_en = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

result = pipeline.invoke({"initial_text": "O tempo, esse senhor que tudo rege, passa inclemente, sem dar tréguas aos que hesitam na senda do propósito. Deixamos rastros de vivências no pó da memória, tecendo a tapeçaria da existência com os fios da escolha e do destino. Mas, para além dos vultos do passado e dos contornos incertos do porvir, reside a urgência do presente, o instante sagrado onde a ação se manifesta. É na efemeridade do agora que reside a força transformadora, capazes de reescrever o curso dos acontecimentos e moldar o que virá a ser"})
print(result)