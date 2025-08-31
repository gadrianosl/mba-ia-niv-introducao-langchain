from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()

long_text = """
O amanhecer tece um ouro pálido através do beco de vidro.
A cidade boceja em um coro de freios e sirenes distantes.
Janelas piscam acordando, uma a uma, como olhos sonolentos.
O vapor da rua se enrola das bocas de lobo, um rio silencioso.
O vapor do café espirala sobre a pálida impressão do jornal.
Pedestres desenham luz nas calçadas, apressados, barulhentos com guarda-chuvas.
Ônibus engolem a manhã com seus bocejos altos.
Um pardal pousa em uma viga de aço, observando a grade urbana.
O metrô suspira em algum lugar subterrâneo, um batimento cardíaco crescente.
O néon ainda brilha nos cantos onde a noite se recusou a se aposentar.
Um ciclista corta através do coro, brilhante com cromo e momento.
A cidade limpa sua garganta, o ar tornando-se um pouco menos elétrico.
Sapatos sibilam no concreto, mil pequenos verbos de chegada.
O amanhecer mantém suas promessas no ritmo silencioso de uma metrópole acordando.
A luz da manhã cascateia através de janelas altas de aço e vidro,
projetando sombras geométricas nas ruas movimentadas abaixo.
O tráfego flui como rios de metal e luz,
enquanto pedestres tecem através das faixas de pedestres com propósito.
Cafeterias exalam calor e o aroma de pão fresco,
enquanto passageiros agarram suas xícaras como talismãs contra o frio.
Vendedores de rua chamam em uma sinfonia de línguas,
suas vozes se misturando com o zumbido distante da construção.
Pombos dançam entre os pés dos trabalhadores apressados,
encontrando migalhas de pães matinais nas calçadas de concreto.
A cidade respira em ritmo com um milhão de batimentos cardíacos,
cada pessoa carregando sonhos e prazos em igual medida.
Arranha-céus alcançam nuvens que flutuam como algodão,
enquanto lá embaixo, trens do metrô ribombam através dos túneis.
Esta orquestra urbana toca do amanhecer até o anoitecer,
uma canção sem fim de ambição, luta e esperança.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=70, 
)

parts = splitter.create_documents([long_text])

# for part in parts:
#     print(part.page_content)
#     print("-"*30)

modelo = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

from langchain.prompts import PromptTemplate

map_template = """Escreva um resumo conciso do seguinte texto em português do Brasil:

{text}

RESUMO CONCISO:"""
map_prompt = PromptTemplate(template=map_template, input_variables=["text"])

combine_template = """Escreva um resumo conciso dos seguintes resumos em português do Brasil:

{text}

RESUMO FINAL:"""
combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])

chain_sumarizar = load_summarize_chain(
    modelo,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    verbose=False
)

resultado = chain_sumarizar.invoke({"input_documents": parts})
print("\nResumo final:")
print(resultado["output_text"])