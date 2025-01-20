import logging
from langchain.docstore.document import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
import boto3
import google.auth
import requests

def get_llm(model: str):
    model = model.lower().strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = os.environ.get(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    try:
        if "gemini" in model:
            model_name = env_value
            credentials, project_id = google.auth.default()
            llm = ChatVertexAI(
                model_name=model_name,
                credentials=credentials,
                project=project_id,
                temperature=0,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
        elif "openai" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0,
            )
        elif "azure" in model:
            model_name, api_endpoint, api_key, api_version = env_value.split(",")
            llm = AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=api_endpoint,
                azure_deployment=model_name,
                api_version=api_version,
                temperature=0,
                max_tokens=None,
                timeout=None,
            )
        elif "anthropic" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatAnthropic(
                api_key=api_key, 
                model=model_name,
                temperature=0,
                timeout=None,
            )
        elif "fireworks" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatFireworks(api_key=api_key, model=model_name)
        elif "groq" in model:
            model_name, base_url, api_key = env_value.split(",")
            llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)
        elif "bedrock" in model:
            model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
            llm = ChatBedrock(
                client=bedrock_client,
                model_id=model_name,
                model_kwargs=dict(temperature=0),
            )
        elif "ollama" in model:
            model_name, base_url = env_value.split(",")
            llm = ChatOllama(base_url=base_url, model=model_name)
        elif "diffbot" in model:
            model_name, api_key = env_value.split(",")
            llm = DiffbotGraphTransformer(
                diffbot_api_key=api_key,
                extract_types=["entities", "facts"],
            )
        else:
            model_name, api_endpoint, api_key = env_value.split(",")
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
            )
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)

    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name

def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list

def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
       Document(
           page_content=document["chunk_doc"].page_content,
           metadata={"chunk_id": [document["chunk_id"]]},
       )
       for document in chunkId_chunkDoc_list
   ]
    return combined_chunk_document_list

def get_wikidata_entity(entity_name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'search': entity_name,
        'language': 'en',
        'format': 'json'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('search'):
            return data['search'][0]
        logging.warning(f"No Wikidata entity found for '{entity_name}'")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Wikidata for '{entity_name}': {e}")
        return None

def enrich_named_entity(entity, entity_type=None):
    entity_info = get_wikidata_entity(entity)
    if entity_info:
        return {
            "entity_name": entity,
            "entity_id": entity_info.get("id"),
            "description": entity_info.get("description"),
            "type": entity_type,
            "wikidata_url": f"https://www.wikidata.org/wiki/{entity_info['id']}"
        }
    return {"entity_name": entity, "type": entity_type, "error": "Not found in Wikidata"}

def evaluate_entities_with_wikidata(graph_document_list):
    enriched_graph_document_list = []
    
    for graph_document in graph_document_list:
        enriched_nodes = []
        for node in graph_document.nodes:
            # Get entity name from id if name not available
            entity_name = getattr(node, 'name', None) or getattr(node, 'id', None)
            if not entity_name:
                logging.warning(f"Node missing both name and id attributes: {node}")
                continue
                
            enriched_node_data = enrich_named_entity(entity_name, node.type)
            
            # Create new node with enriched data
            node_dict = node.dict()
            if "error" not in enriched_node_data:
                node_dict.update({
                    "wikidata_id": enriched_node_data.get("entity_id"),
                    "wikidata_description": enriched_node_data.get("description"),
                    "wikidata_url": enriched_node_data.get("wikidata_url"),
                })
            
            enriched_nodes.append(type(node)(**node_dict))
        
        graph_document.nodes = enriched_nodes
        enriched_graph_document_list.append(graph_document)
    
    return enriched_graph_document_list

async def get_graph_document_list(llm, combined_chunk_document_list, allowedNodes, allowedRelationship):
    if "diffbot_api_key" in dir(llm):
        llm_transformer = llm
    else:
        if "get_name" in dir(llm) and llm.get_name() != "ChatOpenAI" or llm.get_name() != "ChatVertexAI" or llm.get_name() != "AzureChatOpenAI":
            node_properties = False
            relationship_properties = False
        else:
            node_properties = ["description"]
            relationship_properties = ["description"]
        
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
            ignore_tool_usage=True,
        )
    
    if isinstance(llm, DiffbotGraphTransformer):
        graph_document_list = llm_transformer.convert_to_graph_documents(combined_chunk_document_list)
    else:
        graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
    
    # Enrich entities with Wikidata information
    graph_document_list = evaluate_entities_with_wikidata(graph_document_list)
    
    return graph_document_list

async def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    try:
        llm, model_name = get_llm(model)
        combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
        
        if allowedNodes is None or allowedNodes == "":
            allowedNodes = []
        else:
            allowedNodes = allowedNodes.split(',')
            
        if allowedRelationship is None or allowedRelationship == "":
            allowedRelationship = []
        else:
            allowedRelationship = allowedRelationship.split(',')
            
        graph_document_list = await get_graph_document_list(
            llm, combined_chunk_document_list, allowedNodes, allowedRelationship
        )
        return graph_document_list
    except Exception as e:
        err = f"Error during extracting graph with llm: {e}"
        logging.error(err)
        raise