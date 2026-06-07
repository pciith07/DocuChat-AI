from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
data = TextLoader("document_loaders/notes.txt")

template = ChatPromptTemplate.from_messages([
    ("system","you are a AI that summarizes the text"),
    ("human","{data}")
])

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)

docs = data.load()
prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)

print(result.content)

