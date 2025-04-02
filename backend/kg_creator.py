from langchain_core.runnables import  RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver
from langchain_core.prompts import SystemMessagePromptTemplate, PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
import hashlib
from dotenv import load_dotenv

load_dotenv()
graph = Neo4jGraph()


#Chunking the data
loader = TextLoader(file_path="dummytext.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=280, chunk_overlap=24)
documents = text_splitter.split_documents(documents=docs)

#setting up LLM and prompt
llm = OllamaFunctions(model="llama3.1", temperature=0)

PROMPT = SystemMessagePromptTemplate.from_template("""
"I want you to generate a dataset of political arguments from debate platforms, each paired with 1 or more motivations. The dataset should have three columns: 'Argument,' 'Motivation Description,' and 'Max-Neef Need Category.' The Max-Neef categories are limited to: Subsistence, Protection, Affection, Understanding, Participation, Leisure, Creativity, Identity, Freedom.
Each argument should be a unique statement about a political topic (e.g., tax policy, healthcare, climate change), and the motivations must logically align with the argument.
Avoid duplicates and ensure coherence between the argument and its motivations. Below are five examples to guide you:

Argument: 'Tax policy should be expanded to support economic growth'
Motivation Description: 'Wants economic stability for all'
Max-Neef Need Category: Subsistence
                                                   
Argument: 'Climate change needs tighter restrictions to safeguard future generations'
Motivation Description: 'Wants to secure the future'
Max-Neef Need Category: Protection
Motivation Description: 'Desires collective action'
Max-Neef Need Category: Participation
                                                   
Argument: 'Healthcare policies should focus on advance social equity'
Motivation Description: 'Hopes to support the vulnerable'
Max-Neef Need Category: Affection
                                                   
Argument: 'Gun control needs tighter restrictions to protect public safety'
Motivation Description: 'Fears threats to safety'
Max-Neef Need Category: Protection
                                                   
Argument: 'Arts funding should prioritize foster cultural development'
Motivation Description: 'Enjoys cultural enrichment'
Max-Neef Need Category: Leisure
Motivation Description: 'Wants innovative solutions'
Max-Neef Need Category: Creativity

...
                                                                                 
Using these examples as a template, extract information from the input. Ensure each argument is distinct, politically relevant, and paired with motivations that make sense in context. Format you response with the 'Argument,' and 'Motivation Description,'"

""")

FINAL_TIP = HumanMessagePromptTemplate(
    prompt=PromptTemplate.from_template("""
The structure the nodes must obey are as follows:
    Argument: description
    Motivation: description; max neef category
The argument and motivation nodes must be related
Tip: Make sure to answer in the correct format and do not include any explanations.
Use the given format to extract information from the following input: {input}
""")
)

chat_prompt = ChatPromptTemplate.from_messages([PROMPT, FINAL_TIP])


allowed_relations = ["SUPPORTS", "AGAINST"]
allowed_nodes = ["Motivation", "Argument"]
#usar node_properties=["name"] caso necessário
#llm_transformer = LLMGraphTransformer(llm=llm, strict_mode=False, allowed_nodes=allowed_nodes, additional_instructions=additional_context)

#utilizar allowed_nodes gera um grafo mais sucinto, mas pode reduzir a complexidade
llm_transformer = LLMGraphTransformer(llm=llm, prompt=chat_prompt, allowed_nodes=allowed_nodes, node_properties=["description"], strict_mode=False)

#Extrair nodes e relacoes
graph_documents = llm_transformer.convert_to_graph_documents(documents)

##Re-mapear relações
# Dicionário para mapear descrições para IDs únicos
description_to_id = {}

# Dicionário para mapear os IDs antigos para os novos (para atualizar relações)
old_to_new_id = {}

def generate_unique_id(description, type_label):
    """Gera um ID único baseado na descrição e tipo do nó."""
    if description not in description_to_id:
        content_hash = hashlib.md5(description.encode()).hexdigest()[:8]  # Hash curto
        unique_id = f"{type_label}_{content_hash}"  # Exemplo: "Argument_a1b2c3d4"
        description_to_id[description] = unique_id
    
    return description_to_id[description]

# Passo 1: Criar novos IDs para os nós e armazenar a conversão
for doc in graph_documents:
    for node in doc.nodes:
        description = node.properties.get("description", node.id)  # Usa a descrição se existir
        new_id = generate_unique_id(description, node.type)  # Gera o novo ID
        
        old_to_new_id[node.id] = new_id  # Guarda a conversão do ID antigo para o novo
        node.id = new_id  # Atualiza o ID do nó

    # Passo 2: Atualizar as relações para garantir que os IDs estão corretos
    for relation in doc.relationships:
        # Atualizar source e target para os novos IDs
        relation.source.id = old_to_new_id.get(relation.source.id, relation.source.id)
        relation.target.id = old_to_new_id.get(relation.target.id, relation.target.id)


##Upload para o grafo
# Criar ou obter o nó do título do post e conectar argumentos a ele
for doc in graph_documents:
    post_title = doc.source.metadata.get("title", "A Resident's Concerns: The Proposed Airport Near Our Homes")  # Obtém o título do post
    post_id = hashlib.md5(post_title.encode()).hexdigest()[:8]  # Gera um ID único baseado no título

    # Criar o nó do título do post no Neo4j
    graph.query(
        """
        MERGE (p:PostTitle {id: $post_id})
        SET p.title = $post_title
        """,
        {"post_id": post_id, "post_title": post_title}
    )

    # Conectar cada argumento ao nó do título do post
    for node in doc.nodes:
        if "Argument" in node.type:  # Apenas argumentos são ligados ao post
            graph.query(
                """
                MATCH (p:PostTitle {id: $post_id}), (n:Argument {id: $node_id})
                MERGE (n)-[:BELONGS_TO]->(p)
                """,
                {"post_id": post_id, "node_id": node.id}
            )

# Upload do grafo com argumentos e motivações
graph.add_graph_documents(graph_documents=graph_documents, include_source=False)
