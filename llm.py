from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_llm(model="llama3", temperature=0.3):
    return ChatOllama(
        model=model,
        temperature=temperature,
    )

def get_rag_chain(retriever):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Eres un asistente experto en el Reglamento General de Estudiantes "
         "del Tecnológico de Monterrey. Responde únicamente con la información "
         "proporcionada en el contexto. Si la respuesta no se encuentra en el "
         "contexto, indica que no tienes suficiente información.\n\n"
         "Contexto:\n{context}"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
