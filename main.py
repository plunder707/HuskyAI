import os
import logging
import re
import copy
import time
import uuid
import json
import argparse
from typing import List, Dict, Any, Union, Optional
from pprint import pprint
from logging.handlers import RotatingFileHandler

import requests
import numpy as np
import torch

from transformers import (
    Conversation,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline as hf_pipeline,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from haystack.nodes.retriever.web import WebRetriever
from haystack.utils import print_answers, print_documents, print_questions
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import (
    PreProcessor,
    BM25Retriever,
    EmbeddingRetriever,
    PromptNode,
    PromptTemplate,
    JoinDocuments,
    SentenceTransformersRanker,
    MultiModalRetriever,
    FARMReader,
    Shaper,
)
from haystack.pipelines import Pipeline
from haystack.schema import Document
from haystack.agents import Tool
from haystack.nodes.base import BaseComponent
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationMemory, ConversationSummaryMemory

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError

from sentence_transformers import SentenceTransformer


#$Traceloop.init(app_name="Husky", api_key="71f7667b50935835945a3233611c31fc6a838dcda8feb0f1c6cf55e86f7d3680510ec822ad355d6763fb2e32cc0e5141")

os.environ["TRANSFORMERS_OFFLINE_MODE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "debug"


# Set CUDA visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

# Enable oneDNN optimizations for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# Enable CUDA DSA (Direct Stream Access) for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Initialize logging
def setup_logging(log_directory):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "script_logger.log"),
        maxBytes=5 * 1024 * 1024, 
        backupCount=2
    )
    console_handler = logging.StreamHandler()
    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler.setFormatter(logging.Formatter(logging_format))
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def initialize_document_store(args):
    try:
        # Sanitize the index name
        sanitized_index_name = sanitize_index_name(args.index_name)
        args.index_name = sanitized_index_name  # Update args to use the sanitized name
        
        document_store = ElasticsearchDocumentStore(
            host=args.es_host,
            port=args.es_port,
            username=args.es_user,
            password=args.es_pass,
            scheme=args.es_scheme,
            index=sanitized_index_name,
            verify_certs=False,
            return_embedding=True,
            embedding_field="embedding",
            embedding_dim=768,
            search_fields=["*"],
            content_field="content",
            name_field="content",
            custom_mapping=None,
            recreate_index=False,
            similarity="dot_product",
            timeout=9000,
            duplicate_documents="overwrite",
        )
        es_client = Elasticsearch(
            hosts=[f"{args.es_scheme}://{args.es_host}:{args.es_port}"],
            http_auth=(args.es_user, args.es_pass),
            verify_certs=False
        )
        return document_store, es_client
    except Exception as e:
        logging.error(f"Error initializing ElasticsearchDocumentStore: {e}", exc_info=True)
        return None, None

    
class ConversationDB:
    def __init__(self, es_client, index='conversations'):
        self.es = es_client
        self.index = index
        self._ensure_index()
        logging.debug("ConversationDB initialized with index: %s", self.index)

    def _ensure_index(self):
        exists = self.es.indices.exists(index=self.index)
        if not exists:
            self.es.indices.create(index=self.index)
            logging.info("Created Elasticsearch index: %s", self.index)
        else:
            logging.debug("Elasticsearch index already exists: %s", self.index)

    def get_or_create_conversation(self, user_id):
        search_body = {
            "query": {
                "term": {
                    "user_id": user_id
                }
            }
        }
        try:
            resp = self.es.search(index=self.index, body=search_body, size=1)
            if resp['hits']['hits']:
                conv_id = resp['hits']['hits'][0]['_id']
            else:
                conv_id = self.create_conversation(user_id)
        except Exception as e:
            logging.error(f"Error searching for conversation: {e}")
            conv_id = self.create_conversation(user_id)
        return conv_id

    def create_conversation(self, user_id):
        conv_id = str(uuid.uuid4())
        doc = {'user_id': user_id, 'messages': []}
        self.es.index(index=self.index, id=conv_id, body=doc)
        logging.info(f"New conversation created with ID {conv_id}.")
        return conv_id

    def append_message(self, conv_id, text, sender):
        script = {
            "source": """
            if (ctx._source.messages == null) {
                ctx._source.messages = [params.message];
            } else {
                ctx._source.messages.add(params.message);
            }
            """,
            "lang": "painless",
            "params": {
                "message": {"role": sender, "content": text}
            }
        }
        try:
            self.es.update(index=self.index, id=conv_id, body={"script": script})
            logging.info(f"Message appended to conversation {conv_id} successfully.")
        except Exception as e:
            logging.error(f"Error appending message to conversation {conv_id}: {e}")

    def get_conversation(self, conv_id):
        try:
            res = self.es.get(index=self.index, id=conv_id)
            if 'messages' in res['_source']:
                return res['_source']
            else:
                logging.error(f"'messages' field missing in conversation {conv_id}.")
                return {'messages': []}
        except Exception as e:
            logging.error(f"Error getting conversation: {e}")
            return {'messages': []}

    def get_messages(self, conv_id):
        try:
            res = self.es.get(index=self.index, id=conv_id)
            if 'messages' in res['_source']:
                return res['_source']['messages']
            else:
                logging.error(f"'messages' field missing in conversation {conv_id}.")
                return []
        except Exception as e:
            logging.error(f"Error getting messages for conversation {conv_id}: {e}")
            return []

import os
from typing import Optional, List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from PIL import Image
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.schema import Document
from haystack.nodes import PreProcessor
import logging
import torch

import logging
from typing import Optional, List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.schema import Document

import re

def sanitize_index_name(index_name: str) -> str:
    """
    Sanitizes the index name by replacing or removing invalid characters.
    Elasticsearch index names cannot contain certain characters.
    """
    sanitized_name = re.sub(r'[\/\\\s"<>|?,*]', '-', index_name)
    return sanitized_name.lower()  # Elasticsearch index names should be lowercase

def preprocess_and_write_data(
    document_store: ElasticsearchDocumentStore,
    dataset_name: str,
    index_name: Optional[str] = None,
    text_field: str = "text",
    metadata_fields: Optional[List[str]] = None,
    batch_size: int = 1000,
    clear_existing_data: bool = False,
):
    try:
        # Check if the index already exists
        if document_store.get_document_count() > 0:
            # Prompt the user to confirm overwrite
            user_response = input("Data already exists in the document store. Overwrite? (yes/no): ").lower()
            if user_response != 'yes':
                print("Exiting without overwriting data.")
                return False  # Indicate no action was taken due to user choice
            else:
                clear_existing_data = True  # Set clear_existing_data to True if user confirms overwrite

        # If clear_existing_data is True, proceed to clear documents
        if clear_existing_data:
            try:
                document_store.delete_documents()
                print("Existing documents cleared.")
            except Exception as e:
                logging.error(f"Failed to clear existing documents: {e}")
                return False  # Indicate failure to clear documents

        # Load the dataset
        dataset = load_dataset(dataset_name, split="train")
        logging.info(f"Dataset '{dataset_name}' loaded successfully.")
        
        # If index name is not provided, use the dataset name
        if index_name is None:
            index_name = dataset_name

        # Sanitize the index name to avoid issues with Elasticsearch
        sanitized_index_name = sanitize_index_name(index_name)

        # If metadata fields are not provided, use all columns except text field
        if metadata_fields is None:
            metadata_fields = [col for col in dataset.column_names if col != text_field]

        # Initialize the text embedding model (use a larger model if desired)
        text_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Define the custom mapping for Elasticsearch
        custom_mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "content_embedding": {"type": "dense_vector", "dims": text_embedding_model.get_sentence_embedding_dimension()},
                }
            }
        }

        # Add metadata fields to the custom mapping
        for field in metadata_fields:
            if field in dataset.features:
                field_type = dataset.features[field].dtype if not isinstance(dataset.features[field], list) else 'list'
                if field_type == "string":
                    custom_mapping["mappings"]["properties"][field] = {"type": "text"}
                elif field_type == "int64":
                    custom_mapping["mappings"]["properties"][field] = {"type": "long"}
                elif field_type in ["float32", "float64"]:
                    custom_mapping["mappings"]["properties"][field] = {"type": "float"}
                elif field_type == 'list':  # Handle lists appropriately
                    custom_mapping["mappings"]["properties"][field] = {"type": "nested"}
                else:
                    custom_mapping["mappings"]["properties"][field] = {"type": "keyword"}

        # Create or update the index with the sanitized name
        document_store.client.indices.create(index=sanitized_index_name, body=custom_mapping, ignore=400)

        # Process and index the dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            documents = []

            for record in batch:
                if isinstance(record, dict):
                    content = record.get(text_field, "")
                    metadata = {field: record.get(field) for field in metadata_fields if field in record}
                elif isinstance(record, str):
                    content = record
                    metadata = {}
                else:
                    logging.error(f"Unexpected record type: {type(record)}. Skipping.")
                    continue

                # Compute text embeddings
                content_embedding = text_embedding_model.encode(content, convert_to_numpy=True)

                # Create a Haystack Document
                document = Document(content=content, meta=metadata, embedding=content_embedding)
                documents.append(document)

            # Write to Elasticsearch
            document_store.write_documents(documents, index=sanitized_index_name)
            logging.info(f"Processed and indexed batch {i // batch_size + 1}/{len(dataset) // batch_size + 1}")

        return True

    except Exception as e:
        logging.error(f"Error in preprocess_and_write_data: {str(e)}", exc_info=True)
        return False






def get_field_type(field_value):
    if isinstance(field_value, (int, np.integer)):
        return "long"
    elif isinstance(field_value, (float, np.float32, np.float64)):
        return "float"
    elif isinstance(field_value, (str, np.str_)):
        return "text"
    elif isinstance(field_value, (bool, np.bool_)):
        return "boolean"
    elif isinstance(field_value, list):
        return "nested"
    else:
        return "keyword"



class MyConversationalAgent(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
        model_name: str,
        tools: List[Tool],
        prompt_node: PromptNode,
        conversation_db: ConversationDB,
        memory: ConversationMemory,
        conversation_summary_memory: ConversationSummaryMemory,
        document_store: ElasticsearchDocumentStore,
        retriever: MultiModalRetriever,
        user_id: str
    ):
        super().__init__()
        self.model_name = model_name
        self.tools = tools
        self.prompt_node = prompt_node
        self.conversation_db = conversation_db
        self.memory = memory
        self.conversation_summary_memory = conversation_summary_memory
        self.document_store = document_store
        self.retriever = retriever
        self.user_id = user_id

        # Initialize the conversational pipeline
        self.conversation_pipeline = hf_pipeline("conversational", model=self.model_name, device=0 if torch.cuda.is_available() else -1)
        self.user_conversations = {}

    def run(self, query: str, documents: List[Document] = None, **kwargs):
        if not query:
            return {"documents": [{"content": "Please provide a query."}]}, "output_1"

        user_id = self.user_id

        try:
            conv_id = self.conversation_db.get_or_create_conversation(user_id)

            # Get existing messages
            messages = self.conversation_db.get_messages(conv_id)
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

            # Retrieve conversation summary
            conversation_summary = self.conversation_summary_memory.load(user_id).get("summary", "No summary available.")

            # Retrieve relevant documents if not provided
            if documents is None:
                retrieved_documents = self.retriever.retrieve(query=query)
            else:
                retrieved_documents = documents

            # Generate dynamic context using prompt node
            dynamic_context = self.generate_dynamic_context(query, conversation_history, retrieved_documents)

            enriched_query = f"{query}\n\nSummary:\n{conversation_summary}\n\nContext:\n{dynamic_context}"

            if user_id in self.user_conversations:
                conversation = self.user_conversations[user_id]
            else:
                conversation = Conversation()
                self.user_conversations[user_id] = conversation

            conversation.add_user_input(enriched_query)
            self.conversation_pipeline(conversation)
            response_text = conversation.generated_responses[-1] if conversation.generated_responses else "Sorry, I couldn't process that."

            # Update conversation in the database
            self.conversation_db.append_message(conv_id, query, "user")
            self.conversation_db.append_message(conv_id, response_text, "bot")

            updated_history = messages + [{"role": "user", "content": query}, {"role": "bot", "content": response_text}]
            self.memory.save(user_id, updated_history, retrieved_documents, query)
            self.conversation_summary_memory.save(user_id, updated_history, retrieved_documents, query)

            return {"documents": [{"content": response_text}]}, "output_1"

        except Exception as e:
            logging.error(f"Unexpected error in MyConversationalAgent: {e}", exc_info=True)
            return {"documents": [{"content": "An error occurred."}]}, "output_1"

    def run_batch(self, queries: List[str], **kwargs):
        """Batch process queries."""
        responses = []
        for query in queries:
            response, _ = self.run(query, **kwargs)
            responses.append(response)
        return responses, ["output_1"] * len(queries)




import nltk
    
class ConversationMemory:
    def __init__(self):
        self.memory_store = {}

    def load(self, user_id):
        # Load from memory and ensure the value is a dictionary
        conversation_data = self.memory_store.get(user_id, {"context": [], "documents": [], "query": ""})
        
        if not isinstance(conversation_data, dict):
            logging.error(f"Invalid data type in memory for user {user_id}. Expected dict, got {type(conversation_data)}. Resetting to default.")
            # Reset to default dictionary structure if incorrect type is found
            self.memory_store[user_id] = {"context": [], "documents": [], "query": ""}
            conversation_data = self.memory_store[user_id]
        
        logging.debug(f"Loaded conversation memory for user {user_id}: {conversation_data}")
        return conversation_data

    def save(self, user_id, context, documents=None, query=None):
        # Ensure that we only store dictionaries
        if isinstance(context, list) and (documents is None or isinstance(documents, list)) and (query is None or isinstance(query, str)):
            self.memory_store[user_id] = {
                "context": context,
                "documents": documents if documents else [],
                "query": query if query else ""
            }
            logging.debug(f"Saved conversation memory for user {user_id}: {self.memory_store[user_id]}")
        else:
            logging.error(f"Invalid data structure for user {user_id}. Context: {context}, Documents: {documents}, Query: {query}")
            return  # Do not save invalid data





class ConversationSummaryMemory:
    def __init__(
        self,
        prompt_node: PromptNode,
        document_store: ElasticsearchDocumentStore,
        retriever: MultiModalRetriever,
        max_summary_length: int = 200,
    ):
        self.prompt_node = prompt_node
        self.document_store = document_store
        self.retriever = retriever
        self.conversation_summaries = {}
        self.max_summary_length = max_summary_length

    def load(self, user_id):
        summary = self.conversation_summaries.get(user_id)
        return copy.deepcopy(summary) if summary else {"history": [], "summary": "", "documents": [], "query": ""}

    def save(self, user_id, context, documents, query):
        if not isinstance(context, list) or not all(isinstance(exchange, dict) for exchange in context):
            logging.error(f"Invalid context format for user_id {user_id}")
            return

        self.conversation_summaries[user_id] = {
            "history": context,
            "documents": documents,
            "query": query,
        }
        self._update_summary(user_id, context)

    def _update_summary(self, user_id, context):
        transcript = "\n".join(f"{exchange['role']}: {exchange['content']}" for exchange in context)
        summary = self._generate_summary(transcript)
        self.conversation_summaries[user_id]["summary"] = summary

    def _generate_summary(self, transcript):
        if not transcript.strip():
            return "No summary available."

        try:
            # Generate summary using the prompt node
            prompt_template_text = "Summarize the following conversation:\n\n{transcript}\n\nSummary:"
            prompt_template = PromptTemplate(prompt_text=prompt_template_text)
            prompt = prompt_template.fill(transcript=transcript)

            result, _ = self.prompt_node.run(prompt=prompt)
            if result and 'documents' in result and result['documents']:
                return result['documents'][0].content
            else:
                logging.warning("Unexpected response format from PromptNode.")
                return "Summary generation failed."

        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return "Summary generation failed."


class CustomConversationalComponent(BaseComponent):
    outgoing_edges = 1

    def __init__(self, model_name, tools, prompt_node, conversation_db, memory):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tools = tools
        self.prompt_node = prompt_node
        self.conversation_db = conversation_db
        self.memory = memory
        self.conversation_pipeline = hf_pipeline("conversational", model=self.model_name)

    def run(self, query: str, **kwargs):
        user_id = kwargs.get("user_id", "default_user_id")
        conv_id = self.conversation_db.get_or_create_conversation(user_id)

        if not query:
            return {"documents": [{"content": "Please provide a query."}]}, "output_1"

        try:
            conversation = Conversation(messages=query)
            self.conversation_pipeline([conversation])
            response_text = conversation.generated_responses[-1] if conversation.generated_responses else "Sorry, I couldn't process that."

            self.conversation_db.append_message(conv_id, query, "user")
            self.conversation_db.append_message(conv_id, response_text, "bot")

            conversation_context = self.conversation_db.get_conversation(conv_id)["messages"]
            self.memory.save(user_id, conversation_context, [response_text], query)

            return {"documents": [{"content": response_text}]}, "output_1"
        except Exception as e:
            logging.error(f"Error in CustomConversationalComponent: {e}")
            return {"documents": [{"content": "An error occurred while processing your request."}]}, "output_1"

    def _create_transcript(self, context):
        return "\n".join(f"{exchange['role']}: {exchange['content']}" for exchange in context)

    def run_batch(self, queries, **kwargs):
        responses = []
        for query in queries:
            if isinstance(query, str):
                response = self.run(query, **kwargs)
                responses.append(response)
            else:
                logging.error("Query is not a string.")
        return responses, "output_1"

    def _process_response(self, response):
        if isinstance(response, dict) and 'documents' in response:
            documents = response.get('documents', [])
            if documents:
                return documents[0].get('content', 'Default response')
            else:
                logging.error("No documents found in response.")
        else:
            logging.error("Unexpected response format.")
        return {"documents": [{"content": "Default response"}]}


class HuggingFaceConversationalModel(BaseComponent):
    outgoing_edges = 1

    def __init__(self, model_name: str = "microsoft/DialoGPT-large"):
        super().__init__()
        self.model_name = model_name
        self.conversation_pipeline = hf_pipeline("conversational", model=self.model_name, device=0 if torch.cuda.is_available() else -1)
        # Keep track of conversations for each user
        self.user_conversations = {}

    def run(self, query: str, documents: List[Document] = None, **kwargs):
        user_id = kwargs.get("user_id", "default_user_id")

        if not query:
            return {"documents": [{"content": "Please provide a query."}]}, "output_1"

        # Retrieve or create a conversation for the user
        if user_id in self.user_conversations:
            conversation = self.user_conversations[user_id]
        else:
            conversation = Conversation()
            self.user_conversations[user_id] = conversation

        # Add documents content as additional context if provided
        if documents:
            documents_text = "\n".join([doc.content for doc in documents])
            conversation.add_user_input(documents_text)

        # Add user input
        conversation.add_user_input(query)

        self.conversation_pipeline(conversation)
        response_text = conversation.generated_responses[-1] if conversation.generated_responses else "Sorry, I couldn't process that."
        return {"documents": [{"content": response_text}]}, "output_1"

    def run_batch(self, queries: List[str], **kwargs):
        responses = []
        for query in queries:
            response, _ = self.run(query, **kwargs)
            responses.append(response)
        return responses, ["output_1"] * len(queries)




import torch
import torch.nn.functional as F
from typing import List, Dict
from haystack.schema import Document
from haystack.nodes import BaseComponent

class MultiModelAgent(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
        conversation_db: ConversationDB,
        user_id: str,
        conversation_memory: ConversationMemory,
        conversation_summary_memory: ConversationSummaryMemory,
        retriever: MultiModalRetriever,
        models: List[BaseComponent],  # List of models (can be causal LM or seq2seq)
        query_embedding_model,        # Model to generate embeddings for queries and responses
        prompt_node: PromptNode,
        attention_weighting: bool = True,  # Whether to apply attention-based fusion
    ):
        super().__init__()
        self.conversation_db = conversation_db
        self.user_id = user_id
        self.conversation_memory = conversation_memory
        self.conversation_summary_memory = conversation_summary_memory
        self.retriever = retriever
        self.models = models
        self.query_embedding_model = query_embedding_model
        self.prompt_node = prompt_node
        self.attention_weighting = attention_weighting

    def compute_dot_product_similarity(self, query_embedding, response_embedding):
        """Compute the dot product similarity between query and response embeddings."""
        return torch.dot(query_embedding, response_embedding)

    def get_embeddings(self, text: str):
        """Generate embeddings for the text using the query embedding model."""
        embedding = self.query_embedding_model.encode(text, convert_to_tensor=True)
        return embedding

    def apply_attention_weights(self, similarities: torch.Tensor, responses: List[str]):
        """Apply attention weights based on dot-product similarities to fuse responses."""
        attention_weights = F.softmax(similarities, dim=0)  # Normalize to get attention weights
        weighted_responses = [weight * response for weight, response in zip(attention_weights, responses)]
        
        # Combine responses based on weights
        final_response = torch.sum(torch.stack(weighted_responses), dim=0)
        return final_response

    def run(self, query: str, **kwargs):
        if not query:
            return {"documents": [{"content": "The query is empty. Please provide a valid query."}]}, "output_1"

        try:
            conv_id = self.conversation_db.get_or_create_conversation(self.user_id)
            messages = self.conversation_db.get_messages(conv_id)
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

            # Retrieve relevant documents
            retrieved_documents = self.retriever.retrieve(query=query)

            # Get query embedding
            query_embedding = self.get_embeddings(query)

            # Collect model outputs and their embeddings
            model_responses = []
            model_embeddings = []
            for model in self.models:
                response, _ = model.run(
                    query=query,
                    conversation_history=conversation_history,
                    retrieved_documents=retrieved_documents,
                )
                if response and 'documents' in response:
                    response_text = response['documents'][0].content
                    model_responses.append(response_text)
                    response_embedding = self.get_embeddings(response_text)
                    model_embeddings.append(response_embedding)

            # Calculate similarity scores (dot products between query and response embeddings)
            similarity_scores = []
            for response_embedding in model_embeddings:
                similarity = self.compute_dot_product_similarity(query_embedding, response_embedding)
                similarity_scores.append(similarity)

            similarity_scores = torch.tensor(similarity_scores)

            # If attention-weighting is enabled, fuse responses based on attention
            if self.attention_weighting:
                best_response = self.apply_attention_weights(similarity_scores, model_responses)
            else:
                # Simply select the response with the highest similarity
                best_response_index = torch.argmax(similarity_scores)
                best_response = model_responses[best_response_index]

            # Update conversation and memories
            self.conversation_db.append_message(conv_id, query, "user")
            self.conversation_db.append_message(conv_id, best_response, "bot")
            self.conversation_memory.save(self.user_id, messages + [{"role": "user", "content": query}, {"role": "bot", "content": best_response}])
            self.conversation_summary_memory.update_summary(self.user_id, query, best_response, retrieved_documents)

            return {"documents": [{"content": best_response}]}, "output_1"

        except Exception as e:
            logging.error(f"Exception in MultiModelAgent: {str(e)}")
            return {"documents": [{"content": "An error occurred while processing your request."}]}, "output_1"

    def run_batch(self, queries: List[str], **kwargs):
        responses = []
        for query in queries:
            response, _ = self.run(query, **kwargs)
            responses.append(response)
        return responses, ["output_1"] * len(queries)


from typing import List
from haystack.nodes import BaseComponent
from haystack.schema import Document
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

class BlenderbotConversationalAgent(BaseComponent):
    outgoing_edges = 1

    def __init__(self, conversation_memory, model_name: str = "facebook/blenderbot-400M-distill", user_id: str = "default_user_id"):
        super().__init__()
        self.model_name = model_name
        self.conversation_memory = conversation_memory
        self.user_id = user_id
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    def run(self, query: str, documents: List[Document] = None, **kwargs):
        if not query:
            return {"documents": [{"content": "Please provide a query."}]}, "output_1"

        user_id = self.user_id

        # Load conversation history from memory
        conversation_history = self.conversation_memory.load(user_id).get("context", [])

        # Check if the conversation history is valid (should be a list)
        if isinstance(conversation_history, str):
            # If it's a string, we can't use it directly for concatenation, so we wrap it as a list
            logging.error(f"Conversation history for user {user_id} was a string. Converting to a list.")
            conversation_history = [{"role": "assistant", "content": conversation_history}]
        elif not isinstance(conversation_history, list):
            # If it's neither a list nor string, reset it to an empty list
            logging.error(f"Invalid conversation history type: {type(conversation_history)}. Resetting to an empty list.")
            conversation_history = []

        # If documents were provided, add their content to the conversation history
        if documents:
            documents_text = "\n".join([doc.content for doc in documents])
            conversation_history.append({"role": "system", "content": documents_text})

        # Generate response using the Blenderbot model
        response_text = self.generate_response(query, conversation_history)

        # Update conversation history with the user's query and the bot's response
        updated_history = conversation_history + [
            {"role": "user", "content": query},
            {"role": "bot", "content": response_text}
        ]

        # Save the updated conversation history in memory
        self.conversation_memory.save(user_id, updated_history)

        return {"documents": [{"content": response_text}]}, "output_1"

    def run_batch(self, queries: List[str], documents: Union[List[Document], List[List[Document]]] = None, **kwargs):
        if documents is None:
            documents = [[] for _ in range(len(queries))]

        responses = []
        for query, docs in zip(queries, documents):
            result, _ = self.run(query, documents=docs, **kwargs)
            responses.append(result)

        return responses, "output_1"

    def generate_response(self, query, conversation_history):
        # Format the conversation history for input to the model
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

        # Generate response with Blenderbot
        input_ids = self.tokenizer(f"{history_text}\nUser: {query}\nAssistant:", return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_length=150, num_return_sequences=1)
        response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response_text




import torch
import torch.nn.functional as F
from typing import List, Dict
from haystack.schema import Document
from haystack.nodes import BaseComponent

class HybridFusionAgent(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
        models: List[BaseComponent],
        retriever: MultiModalRetriever,
        conversation_memory: ConversationMemory,
        conversation_summary_memory: ConversationSummaryMemory,
        user_id: str,
        query_embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
        attention_weighting: bool = True
    ):
        super().__init__()
        self.models = models
        self.retriever = retriever
        self.attention_weighting = attention_weighting
        self.embedding_model = SentenceTransformer(query_embedding_model)
        self.conversation_memory = conversation_memory
        self.conversation_summary_memory = conversation_summary_memory
        self.user_id = user_id

    def compute_dot_product_similarity(self, query_embedding, response_embedding):
        """Compute the dot product similarity between query and response embeddings."""
        return torch.dot(query_embedding, response_embedding)

    def get_embeddings(self, text: str):
        """Generate embeddings for the text using the query embedding model."""
        embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        return embedding

    def apply_attention_weights(self, similarities: torch.Tensor, responses: List[str]):
        """Apply attention weights based on dot-product similarities to fuse responses."""
        attention_weights = F.softmax(similarities, dim=0)  # Normalize to get attention weights
        weighted_responses = [weight * response for weight, response in zip(attention_weights, responses)]
        
        # Combine responses based on weights
        final_response = torch.sum(torch.stack(weighted_responses), dim=0)
        return final_response

    def run(self, query: str, **kwargs):
        if not query:
            return {"documents": [{"content": "Please provide a query."}]}, "output_1"

        try:
            user_id = self.user_id

            # Retrieve conversation summary
            conversation_summary = self.conversation_summary_memory.load(user_id).get("summary", "No summary available.")

            # Retrieve relevant documents for additional context
            retrieved_documents = self.retriever.retrieve(query=query)
            document_context = "\n".join([doc.content for doc in retrieved_documents])

            # Get query embedding (combine query with conversation summary and document context)
            enriched_query = f"{query}\n\nSummary:\n{conversation_summary}\n\nContext:\n{document_context}"
            query_embedding = self.get_embeddings(enriched_query)

            # Collect model outputs and their embeddings
            model_responses = []
            model_embeddings = []
            for model in self.models:
                response, _ = model.run(query=query, documents=retrieved_documents, user_id=user_id)
                if response and 'documents' in response and response['documents']:
                    response_text = response['documents'][0]['content']
                    model_responses.append(response_text)
                    response_embedding = self.get_embeddings(response_text)
                    model_embeddings.append(response_embedding)
                else:
                    logging.warning(f"Model {model} did not return a valid response.")

            if not model_responses:
                logging.error("No valid responses from models.")
                return {"documents": [{"content": "I'm sorry, I couldn't generate a response."}]}, "output_1"

            # Calculate similarity scores (cosine similarities between query and response embeddings)
            similarity_scores = []
            for response_embedding in model_embeddings:
                similarity = self.compute_dot_product_similarity(query_embedding, response_embedding)
                similarity_scores.append(similarity)

            similarity_scores = torch.tensor(similarity_scores)

            # Apply attention-weighted fusion or select the best response
            if self.attention_weighting:
                final_response = self.apply_attention_weights(similarity_scores, model_responses)
            else:
                # Select the response with the highest similarity score
                best_response_index = torch.argmax(similarity_scores)
                final_response = model_responses[best_response_index]

            return {"documents": [{"content": final_response}]}, "output_1"

        except Exception as e:
            logging.error(f"Exception in HybridFusionAgent: {str(e)}", exc_info=True)
            return {"documents": [{"content": "An error occurred while processing your request."}]}, "output_1"

    def run_batch(self, queries: List[str], **kwargs):
        """Batch process queries."""
        responses = []
        for query in queries:
            response, _ = self.run(query, **kwargs)
            responses.append(response)
        return responses, ["output_1"] * len(queries)




def get_advanced_conversational_pipeline(
    document_store,
    web_retriever: WebRetriever,
    conversation_db: ConversationDB,
    user_id: str,
    conversation_memory: ConversationMemory,
    conversation_summary_memory: ConversationSummaryMemory,
    prompt_node_instance: PromptNode,
):
    # Initialize retrievers
    bm25_retriever = BM25Retriever(
        document_store=document_store,
        top_k=10
    )

    embedding_retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/msmarco-distilbert-base-dot-prod-v3",
        use_gpu=True
    )

    # Initialize the MultiModalRetriever
    # Modify the initialization of the MultiModalRetriever
    retriever_text_to_image = MultiModalRetriever(
        document_store=document_store,
        query_embedding_model="sentence-transformers/all-mpnet-base-v2",  # Text model
        document_embedding_models={
            "text": "sentence-transformers/all-mpnet-base-v2",  # Text embedding model
            "image": "sentence-transformers/clip-ViT-B-32",
        },
        query_type="text",
        embed_meta_fields=["*"],
        top_k=25,
        batch_size=16,
        similarity_function="dot_product",
        progress_bar=True
    )


    # Update embeddings for the MultiModalRetriever if needed
    update_embeddings = False
    if update_embeddings:
        try:
            document_store.update_embeddings(retriever=retriever_text_to_image)
            logging.info("Embeddings updated successfully.")
        except Exception as e:
            logging.error(f"Failed to update embeddings for the MultiModalRetriever: {e}")
    else:
        logging.info("Embeddings update skipped.")

    # Initialize the preprocessor
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        split_by="word",
        split_length=2000,
        split_overlap=20,
        split_respect_sentence_boundary=True
    )

    # Initialize the reader
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # Initialize the ranker
    sentence_ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Initialize the shaper node
    shaper = Shaper(
        func="join_documents",
        inputs={"documents": "documents"},
        outputs=["documents"],
        params={"delimiter": "\n\n"}
    )

    # Initialize WebRetriever and WebTool
    web_retriever = WebRetriever(api_key="", top_k=15, top_search_results=15)
    web_tool = Tool(name="WebSearch", pipeline_or_node=web_retriever, description="Searching the web for additional info")


    # Initialize the agents
    # Blenderbot Conversational Agent
    blenderbot_conversational_agent = BlenderbotConversationalAgent(
        conversation_memory=conversation_memory,
        model_name="facebook/blenderbot-400M-distill",
        user_id=user_id
    )

    # HuggingFace Conversational Model
    hf_conversational_model = HuggingFaceConversationalModel()

    # MyConversationalAgent
    my_conversational_agent = MyConversationalAgent(
        model_name="EleutherAI/gpt-neo-2.7B",
        tools=[web_retriever],
        prompt_node=prompt_node_instance,
        conversation_db=conversation_db,
        memory=conversation_memory,
        conversation_summary_memory=conversation_summary_memory,
        document_store=document_store,
        retriever=retriever_text_to_image,
        user_id=user_id
    )

    # HybridFusionAgent
    models_list = [hf_conversational_model, blenderbot_conversational_agent, my_conversational_agent]
    hybrid_fusion_agent = HybridFusionAgent(
        models=models_list,
        retriever=retriever_text_to_image,
        query_embedding_model="sentence-transformers/all-mpnet-base-v2",
        attention_weighting=True,
        conversation_memory=conversation_memory,
        conversation_summary_memory=conversation_summary_memory,
        user_id=user_id
    )

    # Build the pipeline
    conversational_pipeline = Pipeline()

    # Add retrievers to the pipeline
    conversational_pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    conversational_pipeline.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
    conversational_pipeline.add_node(component=retriever_text_to_image, name="MultiModalRetriever", inputs=["Query"])
    conversational_pipeline.add_node(component=web_retriever, name="WebRetriever", inputs=["Query"])

    # Join the documents from different retrievers
    join_documents_node = JoinDocuments(join_mode="merge")
    conversational_pipeline.add_node(component=join_documents_node, name="JoinDocuments", inputs=["BM25Retriever", "EmbeddingRetriever", "MultiModalRetriever", "WebRetriever"])

    # Preprocess the joined documents
    conversational_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["JoinDocuments"])

    # Use the reader to extract answers from documents
    conversational_pipeline.add_node(component=reader, name="Reader", inputs=["Preprocessor"])

    # Rank the reader's outputs to prioritize the most relevant sections
    conversational_pipeline.add_node(component=sentence_ranker, name="Ranker", inputs=["Reader"])

    # Shape the data for the PromptNode
    conversational_pipeline.add_node(component=shaper, name="Shaper", inputs=["Ranker"])

    # Use the PromptNode to generate a response
    conversational_pipeline.add_node(component=prompt_node_instance, name="PromptNode", inputs=["Shaper"])

    # Add the agents
    conversational_pipeline.add_node(component=blenderbot_conversational_agent, name="BlenderbotConversationalAgent", inputs=["PromptNode"])
    conversational_pipeline.add_node(component=hf_conversational_model, name="HFConversationalModel", inputs=["PromptNode"])
    conversational_pipeline.add_node(component=my_conversational_agent, name="MyConversationalAgent", inputs=["PromptNode"])

    # Use the HybridFusionAgent to combine the responses from all agents
    conversational_pipeline.add_node(component=hybrid_fusion_agent, name="HybridFusionAgent", inputs=["BlenderbotConversationalAgent", "HFConversationalModel", "MyConversationalAgent"])

    return conversational_pipeline




# Import necessary libraries and modules
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Conversation,
    pipeline as hf_pipeline
)
from haystack.nodes import (
    PromptNode,
    PromptTemplate,
    AnswerParser
)
from haystack.nodes.prompt.prompt_model import PromptModel
from haystack.nodes.prompt.invocation_layer import HFLocalInvocationLayer
from haystack.modeling.utils import initialize_device_settings
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List, Union, Optional, Dict, Any

# Custom Invocation Layer for Flan-T5
class FlanT5InvocationLayer(HFLocalInvocationLayer):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-large",
        use_gpu: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            use_gpu=use_gpu,
            devices=devices,
            max_length=max_length,
            **kwargs
        )
        self.max_length = max_length

# Prompt Model using Flan-T5
class FlanT5PromptModel(PromptModel):
    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-large",
        use_gpu: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        max_length: int = 512
    ):
        invocation_layer = FlanT5InvocationLayer(
            model_name_or_path=model_name_or_path,
            use_gpu=use_gpu,
            devices=devices,
            max_length=max_length
        )
        super().__init__(
            model_name_or_path=model_name_or_path,
            invocation_layer=invocation_layer
        )

# Sentence Transformer Embedder for Embeddings
class SentenceTransformerEmbedder:
    def __init__(self, model_name_or_path: str = "sentence-transformers/msmarco-MiniLM-L12-cos-v5"):
        self.model = SentenceTransformer(model_name_or_path)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

# Function to generate custom prompt
def generate_custom_prompt(user_input):
    prompt_template_text = "Please summarize the following user input: {user_input}"
    return prompt_template_text.format(user_input=user_input)

# Function to check model availability
def is_model_available(model_name):
    try:
        # Attempt to load the model
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return True
    except Exception:
        return False

# Function to summarize text using Flan-T5
def summarize_text(text):
    summarizer = hf_pipeline("summarization", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)
    return summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

# Adapter class for Hugging Face conversational models
class HuggingFaceAdapter:
    def __init__(self, model_identifier: str):
        # Initialize the conversational pipeline with the specified model
        self.model_identifier = model_identifier
        self.conversational_pipeline = hf_pipeline("conversational", model=self.model_identifier, device=0 if torch.cuda.is_available() else -1)
        
    def generate_response(self, query: str, conv_db, user_id: str) -> Dict[str, Any]:
        # Get or create conversation ID
        conv_id = conv_db.get_or_create_conversation(user_id)

        # Initialize conversation with existing messages
        existing_messages = conv_db.get_messages(conv_id)
        conversation = Conversation()
        for msg in existing_messages:
            if msg['role'] == 'user':
                conversation.add_user_input(msg['content'])
            else:
                conversation.append_response(msg['content'])

        # Add the user's query as a new message
        conversation.add_user_input(query)

        # Generate a response using the conversational pipeline
        response = self.conversational_pipeline(conversation)

        # Extract and return the latest response generated for the conversation
        response_text = response.generated_responses[-1] if response.generated_responses else "Sorry, I couldn't process that."

        # Save the conversation back to the database
        conv_db.append_message(conv_id, query, "user")
        conv_db.append_message(conv_id, response_text, "bot")

        return {"documents": [{"content": response_text}]}

# Function to concatenate response texts
def concatenate_response_texts(detailed_response):
    # Example implementation
    return " ".join([response['content'] for response in detailed_response])

import re
from typing import List

class LaTeXOutputParser:
    def __init__(self):
        # Define regex pattern to extract content after 'Answer:'
        self.pattern = r'Answer:\s*(.*)'

    def parse(self, output: List[str]) -> str:
        """
        Parses the model output and wraps the answer content in LaTeX delimiters.

        Args:
            output (List[str]): The raw output from the model.

        Returns:
            str: The parsed content formatted in LaTeX.
        """
        if isinstance(output, list):
            output_text = output[0]  # Assume single output for simplicity
        else:
            output_text = output

        match = re.search(self.pattern, output_text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Wrap content in LaTeX delimiters for inline or block math
            if len(content.splitlines()) > 1:  # Multiline for block math
                latex_content = f"$$\n{content}\n$$"
            else:  # Single line for inline math
                latex_content = f"\\( {content} \\)"
        else:
            latex_content = "No LaTeX content found in the response."

        return latex_content

class MemoryAugmentedPromptNode(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
        prompt_node: PromptNode,
        conversation_memory: ConversationMemory,
        conversation_summary_memory: Optional[ConversationSummaryMemory],
        document_store: ElasticsearchDocumentStore,
        query_embedding_model: SentenceTransformer,
        retriever: MultiModalRetriever,
        user_id: str,
        **kwargs
    ):
        super().__init__()
        self.prompt_node = prompt_node
        self.conversation_memory = conversation_memory
        self.conversation_summary_memory = conversation_summary_memory
        self.document_store = document_store
        self.query_embedding_model = query_embedding_model
        self.retriever = retriever
        self.user_id = user_id

    def run(self, query: str = None, documents: List[Document] = None, **kwargs):
        try:
            # Retrieve conversation history and summary
            conversation_data = self.conversation_memory.load(self.user_id)
            if not isinstance(conversation_data, dict):
                logging.error(f"Invalid conversation data type: {type(conversation_data)}. Expected dict.")
                conversation_data = {"context": [], "documents": [], "query": ""}
            
            conversation_history = conversation_data.get("context", [])
            if not isinstance(conversation_history, list):
                logging.error(f"Invalid conversation history type: {type(conversation_history)}. Expected list.")
                conversation_history = []

            conversation_history_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in conversation_history]
            )
            conversation_summary = self.conversation_summary_memory.load(self.user_id).get("summary", "No summary available.")

            # Retrieve relevant documents if not provided
            if documents is None or len(documents) == 0:
                retrieved_documents = self.retriever.retrieve(query=query)
                documents = retrieved_documents

            # Prepare relevant documents text
            relevant_documents = "\n".join([doc.content for doc in documents]) if documents else "No relevant documents available."

            # Prepare prompt variables
            prompt_kwargs = {
                "query": query,
                "summary": conversation_summary,
                "transcript": conversation_history_text,
                "documents": relevant_documents,
            }

            # Use the prompt node to generate a response
            result = self.prompt_node.prompt(**prompt_kwargs)

            # Extract the response content
            response = result[0]

            # Log the response
            logging.debug(f"Generated response for user {self.user_id}: {response}")

            # Return the response in the expected format
            return {"documents": [Document(content=response)]}, "output_1"

        except Exception as e:
            logging.error(f"Error in MemoryAugmentedPromptNode run method: {e}", exc_info=True)
            return {"documents": [Document(content="An error occurred during response generation. Please try again.")]}, "output_1"

    def run_batch(self, queries: List[str] = None, documents: List[List[Document]] = None, **kwargs):
        results = []
        for i, query in enumerate(queries):
            # Get the corresponding documents for each query, if provided
            docs = documents[i] if documents else None
            res, _ = self.run(query=query, documents=docs, **kwargs)
            results.append(res)
        return results, "output_1"





from haystack.nodes import PromptNode, PromptTemplate

def get_prompt_node(
    user_id: str,
    conversation_memory: ConversationMemory,
    conversation_summary_memory: Optional[ConversationSummaryMemory],
    document_store: ElasticsearchDocumentStore,
    query_embedding_model: SentenceTransformer,
    retriever: MultiModalRetriever,
    use_gpu: bool = True,
    devices: Optional[List[Union[str, torch.device]]] = None,
) -> MemoryAugmentedPromptNode:
    """
    Initializes and returns a MemoryAugmentedPromptNode configured with the user's ID and memory,
    supporting LaTeX-formatted outputs.
    """

    # Define the custom prompt template
    prompt_template_text = (
        "You are an advanced AI assistant. Provide detailed and comprehensive answers. "
        "Use LaTeX formatting for mathematical expressions in your answers.\n\n"
        "Conversation Summary:\n{summary}\n\n"
        "Conversation History:\n{transcript}\n\n"
        "Relevant Context from Documents:\n{documents}\n\n"
        "User Query: {query}\n\n"
        "Provide a thorough and detailed answer to the user's query."
    )


    prompt_template = PromptTemplate(
        prompt=prompt_template_text,
        output_parser=LaTeXOutputParser()
    )

    # Initialize the PromptNode with the desired model and the custom template
    prompt_node = PromptNode(
        model_name_or_path="google/flan-t5-large",  # You can replace this with any other supported model
        default_prompt_template=prompt_template,
        use_gpu=use_gpu
    )

    # Instantiate the MemoryAugmentedPromptNode with the PromptNode instance
    memory_augmented_prompt_node = MemoryAugmentedPromptNode(
        prompt_node=prompt_node,
        conversation_memory=conversation_memory,
        conversation_summary_memory=conversation_summary_memory,
        document_store=document_store,
        query_embedding_model=query_embedding_model,
        retriever=retriever,
        user_id=user_id
    )

    return memory_augmented_prompt_node







from haystack import BaseComponent





def get_conversational_agent(
    user_id: str,
    document_store: ElasticsearchDocumentStore,
    conv_db: ConversationDB,
    conversation_memory: ConversationMemory,
    conversation_summary_memory: ConversationSummaryMemory,
    web_retriever: WebRetriever,
    use_gpu: bool = True,
    devices: List[Union[str, torch.device]] = None,
) -> ConversationalAgent:
    """
    Initializes and returns a ConversationalAgent configured to use document_store and conv_db for memory
    and retrieval.

    Args:
        user_id (str): A unique identifier for the user.
        document_store (ElasticsearchDocumentStore): The document store for retrieving relevant context.
        conv_db (ConversationDB): The conversation database for tracking user conversations.
        conversation_memory (ConversationMemory): Memory object to manage short-term conversation history.
        conversation_summary_memory (ConversationSummaryMemory): Memory object to manage long-term conversation summaries.
        web_retriever (WebRetriever): A retriever to get external data when needed.
        use_gpu (bool): Flag to indicate whether to use GPU for processing.
        devices (Optional[List[Union[str, torch.device]]]): Devices to use for processing.

    Returns:
        ConversationalAgent: Configured conversational agent.
    """

    # Define the prompt template for generating responses based on conversation and documents
    prompt_template_text = (
        "You are an advanced AI assistant. Provide detailed and comprehensive answers. "
        "Use LaTeX formatting for mathematical expressions in your answers.\n\n"
        "Conversation Summary:\n{summary}\n\n"
        "Conversation History:\n{transcript}\n\n"
        "Relevant Context from Documents:\n{documents}\n\n"
        "User Query: {query}\n\n"
        "Provide a thorough and detailed answer to the user's query."
    )


    prompt_template = PromptTemplate(prompt_text=prompt_template_text)

    # Initialize the prompt model with Flan-T5-XXL or a similar large model
    prompt_model = FlanT5PromptModel(
        model_name_or_path="google/flan-t5-large",
        use_gpu=use_gpu,
        devices=devices
    )

    class DocumentAugmentedConversationalAgent(ConversationalAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.document_store = document_store  # Elasticsearch document store for retrieval
            self.conv_db = conv_db  # Conversation database to track dialogue
            self.conversation_memory = conversation_memory  # Memory for storing conversation history
            self.conversation_summary_memory = conversation_summary_memory  # Memory for conversation summaries
            self.prompt_node = kwargs.get('prompt_node')  # The prompt node to generate responses
            self.user_id = user_id  # Explicitly setting user_id for the agent

        def run(self, query: str, **kwargs):
            """
            Executes a conversation turn by retrieving relevant documents and accessing conversation history.

            Args:
                query (str): User's input query.
                **kwargs: Additional keyword arguments.

            Returns:
                Dict[str, Any]: Generated response.
            """
            # Ensure user_id is always set correctly
            user_id = kwargs.get("user_id", self.user_id)

            if not query:
                return {"documents": [{"content": "Please provide a query."}]}, "output_1"

            # Step 1: Retrieve conversation history from the conversation memory (if available)
            conversation_history = self.conversation_memory.load(user_id).get("context", [])
            conversation_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

            # Step 2: Retrieve relevant documents from the document store (e.g., Elasticsearch)
            retrieved_documents = self.document_store.query(query=query)
            document_content = "\n".join([doc.content for doc in retrieved_documents])

            # Step 3: Load conversation summary from conversation summary memory (if available)
            conversation_summary = self.conversation_summary_memory.load(user_id).get("summary", "No summary available.")

            # Step 4: Generate the prompt using the conversation history, summary, and documents
            prompt_kwargs = {
                "query": query,
                "summary": conversation_summary,
                "transcript": conversation_history_text,
                "documents": document_content or "No relevant documents found."
            }

            prompt = self.prompt_node.default_prompt_template.fill(**prompt_kwargs)

            # Generate response using the prompt model
            output = self.prompt_node.model.generate(
                prompt=prompt,
                max_new_tokens=512,
                num_return_sequences=1,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            response_text = output[0].strip() if output else "Sorry, I couldn't process your query."

            # Step 5: Save the query and response to the conversation database for future context
            conv_id = self.conv_db.get_or_create_conversation(user_id)
            self.conv_db.append_message(conv_id, query, "user")
            self.conv_db.append_message(conv_id, response_text, "bot")

            # Step 6: Update the conversation memory
            updated_history = conversation_history + [
                {"role": "user", "content": query},
                {"role": "bot", "content": response_text}
            ]
            self.conversation_memory.save(user_id, updated_history, retrieved_documents, query)

            # Step 7: Update the conversation summary memory
            self.conversation_summary_memory.save(user_id, updated_history, retrieved_documents, query)

            return {"documents": [{"content": response_text}]}, "output_1"

    # Instantiate the conversational agent with memory, prompt node, and retrieval capabilities
    document_augmented_conversational_agent = DocumentAugmentedConversationalAgent(
        prompt_node=PromptNode(model=prompt_model, default_prompt_template=prompt_template),
        tools=[web_retriever],
        conversation_memory=conversation_memory,
        conversation_summary_memory=conversation_summary_memory
    )

    return document_augmented_conversational_agent


class PromptNode(BaseComponent):
    outgoing_edges = 1

    def __init__(self, prompt_model: PromptModel, default_prompt_template: PromptTemplate, **kwargs):
        super().__init__()
        self.model = prompt_model
        self.default_prompt_template = default_prompt_template

    def get_prompt_template(self, template_name=None):
        return self.default_prompt_template

    def run(self, query: str = None, documents: List[Document] = None, **kwargs):
        try:
            prompt_kwargs = {
                "query": query,
                "transcript": kwargs.get("transcript", ""),
                "summary": kwargs.get("summary", "")
            }

            if documents:
                context = "\n".join([doc.content for doc in documents])
                prompt_kwargs["documents"] = context

            prompt = self.default_prompt_template.fill(**prompt_kwargs)

            if not isinstance(prompt, str):
                raise ValueError("Generated prompt is not a valid string.")

            output = self.model.generate(prompt=prompt, max_new_tokens=512, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
            response = output[0].strip()

            logging.debug(f"Generated response: {response}")
            return {"documents": [Document(content=response)]}, "output_1"

        except Exception as e:
            logging.error(f"Error in PromptNode run method: {e}")
            return {"documents": [Document(content="I apologize, an error occurred during response generation. Let me try again.")]}, "output_1"

    def run_batch(self, queries: List[str], transcripts: List[str] = None, summaries: List[str] = None, documents: Union[List[Document], List[List[Document]]] = None, **kwargs):
        try:
            if transcripts is None:
                transcripts = [""] * len(queries)
            if summaries is None:
                summaries = [""] * len(queries)
            if documents is None:
                documents = [[] for _ in range(len(queries))]

            results = []
            for query, transcript, summary, doc_list in zip(queries, transcripts, summaries, documents):
                result = self.run(query=query, transcript=transcript, summary=summary, documents=doc_list, **kwargs)
                results.append(result)

            return results, "output_1"

        except Exception as e:
            logging.error(f"Error in PromptNode run_batch method: {e}")
            return [{"documents": [Document(content="I apologize, an error occurred during batch processing. Let me try again.")]} for _ in queries], "output_1"





def get_hf_pipeline(task: str):
    """
    Initializes and returns a Hugging Face pipeline for the specified task using appropriate models.
    """
    # Mapping of task names to Hugging Face model identifiers
    task_to_model_mapping = {
        "summarization": "google/flan-t5-base",
        "question-answering": "google/flan-t5-base",
        "conversational": "facebook/blenderbot-3B",
    }
    
    model_identifier = task_to_model_mapping.get(task)
    if not model_identifier:
        raise ValueError(f"No model found for task '{task}'. Please define a model identifier for this task.")
    
    # Initialize the pipeline for the specified task
    device = 0 if torch.cuda.is_available() else -1
    hf_pipeline_instance = hf_pipeline(task, model=model_identifier, device=device)
    
    return hf_pipeline_instance

def is_complex_query(query: str) -> bool:
    """
    Determines if the query is complex based on the presence of certain keywords.
    """
    complex_keywords = {
        "who", "what", "when", "where", "why", "how", "is", "are",
        "was", "were", "am", "been", "do", "does", "did", "don't",
        "doesn't", "didn't"
    }
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in complex_keywords)

def get_dynamic_user_id() -> str:
    """
    Generates a unique user ID.
    """
    return str(uuid.uuid4())

import re

def extract_response_content(pipeline_output):
    """
    Extracts the bot's response from the pipeline output.
    """
    if pipeline_output and 'answers' in pipeline_output and pipeline_output['answers']:
        return pipeline_output['answers'][0].answer.strip()
    elif pipeline_output and 'generated_text' in pipeline_output:
        return pipeline_output['generated_text'].strip()
    elif pipeline_output and 'generated_responses' in pipeline_output:
        return pipeline_output['generated_responses'][-1].strip()
    else:
        return "I'm sorry, I couldn't generate a response."


def validate_documents(documents):
    for doc in documents:
        if not isinstance(doc, Document):
            logging.error(f"Invalid document type: {type(doc)}. Expected Document object.")
            raise ValueError(f"Invalid document type: {type(doc)}. Expected Document object.")
        if not hasattr(doc, 'content'):
            logging.error("Document is missing 'content' attribute")
            raise ValueError("Document is missing 'content' attribute")

def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        if isinstance(doc, str):
            processed_docs.append(Document(content=doc, content_type="text"))
        elif isinstance(doc, Document):
            processed_docs.append(doc)
        else:
            logging.warning(f"Unexpected document type: {type(doc)}. Skipping this document.")
    return processed_docs

def get_tools_info() -> Dict[str, str]:
    """
    Returns information about available tools.
    """
    # Implement logic to fetch or define tool names and descriptions
    return {
        "WebSearch": "Searches the web for additional information.",
        # Add other tools and their descriptions as needed
    }


from haystack import BaseComponent
from haystack.nodes import PromptNode
from haystack.agents.memory import ConversationMemory
from haystack.schema import Answer
from typing import List, Dict, Any, Optional

class ConversationalAgent(BaseComponent):
    outgoing_edges = 1

    def __init__(
        self,
        prompt_node: PromptNode,
        tools: Optional[List[Any]] = None,
        memory: Optional[ConversationMemory] = None,
        **kwargs
    ):
        super().__init__()
        self.prompt_node = prompt_node
        self.tools = tools if tools else []
        self.memory = memory if memory else ConversationMemory()

    def run(self, query: str, **kwargs) -> tuple[Dict[str, Any], str]:
        user_id = kwargs.get("user_id", "default_user_id")

        if not query:
            return {"answers": [Answer(answer="Please provide a query.")]}, "output_1"

        # Load the conversation context
        conversation_context = self.memory.load(user_id)
        history = conversation_context.get('history', [])
        transcript = "\n".join([msg['content'] for msg in history])

        # Generate response using the prompt node
        prompt_kwargs = {
            "query": query,
            "transcript": transcript
        }
        result = self.prompt_node.run(**prompt_kwargs)

        # Extract the response content
        response_content = extract_response_content(result)

        # Update conversation history
        history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response_content}
        ])
        self.memory.save(user_id, context=history)

        return {"answers": [Answer(answer=response_content)]}, "output_1"

    def run_batch(self, queries: List[str], **kwargs) -> tuple[List[Dict[str, Any]], str]:
        responses = []
        for query in queries:
            response, _ = self.run(query, **kwargs)
            responses.append(response)
        return responses, "output_1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_host", type=str, default="localhost")
    parser.add_argument("--es_port", type=int, default=9200)
    parser.add_argument('--es_scheme', type=str, default='http')
    parser.add_argument('--es_user', type=str, default='elastic')
    parser.add_argument('--es_pass', type=str, required=True)
    parser.add_argument("--index_name", type=str, default="document_index")
    parser.add_argument("--dataset_name", type=str, required=False)
    parser.add_argument("--log_directory", type=str, default="/home/plunder/logs")
    args = parser.parse_args()

    setup_logging(args.log_directory)
    
    document_store, es_client = initialize_document_store(args)

    if not document_store or not es_client:
        logging.error("Failed to initialize the document store or Elasticsearch client. Exiting.")
        return
    
    # Preprocess and write data to the document store
    success = preprocess_and_write_data(
        document_store=document_store,
        dataset_name=args.dataset_name,
        batch_size=1000,
        text_field="text",
        metadata_fields=None
    )
    
    if not success:
        logging.info("Data preprocessing and indexing step skipped.")
    else:
        logging.info("Data preprocessing and indexing completed successfully.")

    # Update embeddings before using the retriever
    #document_store.update_embeddings(retriever=retriever_text_to_image)

    # Modify the initialization of the MultiModalRetriever
    retriever_text_to_image = MultiModalRetriever(
        document_store=document_store,
        query_embedding_model="sentence-transformers/all-mpnet-base-v2",  # Text model
        document_embedding_models={
            "text": "sentence-transformers/all-mpnet-base-v2",  # Text embedding model
            "image": "sentence-transformers/clip-ViT-B-32",  # CLIP model for image embeddings
        },
        query_type="text",
        embed_meta_fields=["*"],
        top_k=25,
        batch_size=16,
        similarity_function="dot_product",
        progress_bar=True
    )

    web_retriever = WebRetriever(api_key="")
    web_tool = Tool(name="WebSearch", pipeline_or_node=web_retriever, description="Searching the web for additional info")

    # Generate a unique user ID
    user_id = get_dynamic_user_id()
    logging.debug("Generated dynamic user ID: %s", user_id)

    # Initialize the conversation database
    conv_db = ConversationDB(es_client=es_client, index="conversations")

    # Initialize ConversationMemory
    conversation_memory = ConversationMemory()

    # Initialize the FlanT5InvocationLayer before calling get_prompt_node
    invocation_layer = FlanT5InvocationLayer(
        model_name_or_path="google/flan-t5-base",
        use_gpu=True,  # or False depending on your configuration
        devices=[torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    )

    # Initialize the PromptNode
    prompt_template_text = (
        "You are an advanced AI assistant. Provide detailed and comprehensive answers. "
        "Use LaTeX formatting for mathematical expressions in your answers.\n\n"
        "Conversation Summary:\n{summary}\n\n"
        "Conversation History:\n{transcript}\n\n"
        "Relevant Context from Documents:\n{documents}\n\n"
        "User Query: {query}\n\n"
        "Provide a thorough and detailed answer to the user's query."
    )

    prompt_template = PromptTemplate(
        prompt=prompt_template_text,
        output_parser=LaTeXOutputParser()
    )

    prompt_node = PromptNode(
        model_name_or_path="google/flan-t5-large",
        default_prompt_template=prompt_template,
        use_gpu=True
    )

    # Initialize ConversationSummaryMemory
    conversation_summary_memory = ConversationSummaryMemory(
        prompt_node=prompt_node,
        document_store=document_store,
        retriever=retriever_text_to_image
    )

    # Initialize MemoryAugmentedPromptNode
    memory_augmented_prompt_node = MemoryAugmentedPromptNode(
        prompt_node=prompt_node,
        conversation_memory=conversation_memory,
        conversation_summary_memory=conversation_summary_memory,
        document_store=document_store,
        query_embedding_model=SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
        retriever=retriever_text_to_image,
        user_id=user_id
    )

    # Use memory_augmented_prompt_node in the pipeline
    conversational_pipeline = get_advanced_conversational_pipeline(
        document_store=document_store,
        web_retriever=web_retriever,
        conversation_db=conv_db,
        user_id=user_id,
        conversation_memory=conversation_memory,
        conversation_summary_memory=conversation_summary_memory,
        prompt_node_instance=memory_augmented_prompt_node  # Use the correct instance
    )


    # Debugging: Draw the pipeline structure
    print(conversational_pipeline.draw(path="/mnt/h/pipeline_diagram.png"))

    input("Press Enter to continue...")

    print("Chatbot initialized. Type 'exit' to end the conversation.")

    # Check if a conversation exists for the user, otherwise create a new one
    conv_id = conv_db.get_or_create_conversation(user_id)
    if not conv_id:
        conv_id = conv_db.create_conversation(user_id)
        print(f"New conversation started with ID {conv_id}.")
    else:
        print(f"Existing conversation found with ID {conv_id}.")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Ending conversation. Goodbye!")
                break

            logging.debug(f"User input received: {user_input}")
            pipeline_output = conversational_pipeline.run(
                query=user_input
            )

            bot_response = extract_response_content(pipeline_output)
            print("Bot:", bot_response)

            conv_db.append_message(conv_id, user_input, "user")
            conv_db.append_message(conv_id, bot_response, "bot")

            # Update conversation summary memory with retrieved documents
            conversation_summary_memory.save(user_id, 
                                            conv_db.get_messages(conv_id),
                                            pipeline_output.get("documents", []),
                                            user_input)

        except KeyboardInterrupt:
            print("\nConversation interrupted by user. Exiting.")
            break
        except Exception as e:
            logging.error("An error occurred: %s", e, exc_info=True)
            break

if __name__ == "__main__":
    main()
