from llama_index.core import Settings

# load the ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from BCEmbedding.tools.llama_index import BCERerank
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SimpleNodeParser 


from typing import List

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate

# initialize client, setting path to save data
chroma_client = chromadb.PersistentClient()

class OllamaClient():

    def __init__(self, large_language_model="llama3", embedding_model="bce",response_mode="simple_summarize"):
        self.messages = []

        self.large_language_model = large_language_model
        self.embedding_model = embedding_model
        self.response_mode = response_mode

        self.init_ollama_llm(self.large_language_model)
        # self.init_embeded_moedel(self.embedding_model)
        # self.embed_data(self.embedding_model)
        self.index = self.load_data(embedding_model)
        self.reranker_model = self.rank_model_init()
        self.template1,self.template2=self.set_prompt(response_mode)
        self.query_engine = self.init_query(response_mode,self.reranker_model,self.template1,self.template2)




    def clear_history(self):
        self.messages.clear()

    def append_history(self, message):
        self.messages.append(message)

    def init_ollama_llm(self, llm_type):
        # connect with the ollama server
        llm_llama = Ollama(model=llm_type, request_timeout=600, temperature=0.1, device='cuda')
        Settings.llm = llm_llama
        print("ollama connect to {}".format(llm_type))
        return llm_type

    def init_embeded_moedel(self, embeded_type):
        # connect with the ollama server
        if embeded_type == "llama3":
            embedding_model = OllamaEmbedding(model_name=embeded_type,ollama_additional_kwargs={"mirostat": 0}) 
        elif embeded_type == "bce":
            embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 256, 'device': 'cuda'}
            embedding_model = HuggingFaceEmbedding(**embed_args)
        else:
            embedding_model = OllamaEmbedding(model_name=embeded_type,ollama_additional_kwargs={"mirostat": 0})

        Settings.embed_model = embedding_model
        return embedding_model

    def rank_model_init(self):
        reranker_args = {'model': 'maidalun1020/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda'}
        reranker_model = BCERerank(**reranker_args)
        return reranker_model

    def set_prompt(self, type):
        if type =="simple_summarize":
            template = (
                "You are hellpful, respectful and honest video transcode assistant and very faimilay with ffmpge, and expecially good at MA35D AMA(AMD multimidia accelerator) device encode/decode/transcode.\n"
                "Context information from multiple sources is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the information from multiple sources and not prior knowledge\n"
                "please read the above context information carefully. and anwer the question.\n"
                "if the question is not releate with video process, just say it is not releated with my knowledge base.\n"
                "if you don't know the answer, just say that I don't know.\n"
                "Answers need to be precise and concise.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_template1 = PromptTemplate(template)
            qa_template2 = ""
        
        elif type == "refine":
            template = (
                "You are hellpful, respectful and honest video transcode assistant and very faimilay with ffmpge, and expecially good at MA35D AMA(AMD multimidia accelerator) device encode/decode/transcode.\n"
                "Context information from multiple sources is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the information from multiple sources and not prior knowledge\n"
                "please read the above context information carefully. and anwer the question.\n"
                "if the question is not releate with video process, just say it is not releated with my knowledge base.\n"
                "if you don't know the answer, just say that I don't know.\n"
                "Answers need to be precise and concise.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_template1  = PromptTemplate(template)

            template = (
                "The original query is as follows: {query_str}.\n"
                "We have provided an existing answer: {existing_answer}.\n"
                "We have the opportunity to refine the existing answer (only if needed) with some more context below.\n"
                "-------------\n"
                "{context_msg}\n"
                "-------------\n"
                "Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.\n"
                "if the question is 'who are you' , just say I am a video expert.\n"
                "Answers need to be precise and concise.\n"
                "Refined Answer: "
            )
            qa_template2 = PromptTemplate(template)

        elif type =="tree_summarize":
            template = (
                "You are a Video ffmpeg & gstreamer technolodge expert and expecially good at MA35D AMA(AMD multimidia accelerator) device encode/decode/scale/transcode.\n"
                "Context information from multiple sources is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the information from multiple sources and not prior knowledge, please read the sources carefully.\n"
                "if the question is not releate with the RDMA, just say it is not releated with my knowledge base.\n"
                "if you don't know the answer, just say that I don't know.\n"
                "if the question is 'who are you' , just say I am a FPGA and RDMA expert.\n"
                "Answers need to be precise and concise.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_template1 = PromptTemplate(template)
            qa_template2 = ""
        else:
            template = (
                "You are a Video ffmpeg & gstreamer technolodge expert and expecially good at MA35D AMA(AMD multimidia accelerator) device encode/decode/scale/transcode.\n"
                "Context information from multiple sources is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the information from multiple sources and not prior knowledge, please read the sources carefully.\n"
                "if the question is not releate with the RDMA, just say it is not releated with my knowledge base.\n"
                "if you don't know the answer, just say that I don't know.\n"
                "if the question is 'who are you' , just say I am a FPGA and RDMA expert.\n"
                "Answers need to be precise and concise.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_template1 = PromptTemplate(template)
            qa_template2 = ""
        
        return qa_template1,qa_template2


    def embed_data(self, embedding_type):

        if embedding_type == "llama3":
            base_name = "ma35_rag_base"
            embedding_model = OllamaEmbedding(model_name="llama3_8b",ollama_additional_kwargs={"mirostat": 0}) #base_url="http://localhost:11434"
        elif embedding_type == "bce":
            base_name = "ma35_rag_base_bce"
            embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 256, 'device': 'cuda'}
            embedding_model = HuggingFaceEmbedding(**embed_args)
        else:
            print("embedding model not correct. default llama3\n")
            embedding_model = OllamaEmbedding(model_name="llama3_8b",ollama_additional_kwargs={"mirostat": 0}) #base_url="http://localhost:11434"

        Settings.embed_model = embedding_model
            
        documents = SimpleWebPageReader(html_to_text=True).load_data(
            [
            "https://amd.github.io/ama-sdk/v1.1.2/index.html",
            "https://amd.github.io/ama-sdk/v1.1.2/getting_started_on_prem.html",
            "https://amd.github.io/ama-sdk/v1.1.2/virtualization.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/ffmpeg/tutorials.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/ffmpeg/quality_analysis.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/ffmpeg/filters.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/gstreamer/tutorials.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/gstreamer/filters.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/gstreamer/xcompositor.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/gstreamer/xabrladder.html",
            "https://amd.github.io/ama-sdk/v1.1.2/examples/xma/xma_apps.html",
            "https://amd.github.io/ama-sdk/v1.1.2/specs_and_features.html",
            "https://amd.github.io/ama-sdk/v1.1.2/package_feed.html",
            "https://amd.github.io/ama-sdk/v1.1.2/using_ffmpeg.html",
            "https://amd.github.io/ama-sdk/v1.1.2/using_gstreamer.html",
            "https://amd.github.io/ama-sdk/v1.1.2/unified_logging.html",
            "https://amd.github.io/ama-sdk/v1.1.2/tuning_video_quality.html",
            "https://amd.github.io/ama-sdk/v1.1.2/tuning_pipeline_latency.html",
            "https://amd.github.io/ama-sdk/v1.1.2/managing_compute_resources.html",
            "https://amd.github.io/ama-sdk/v1.1.2/c_apis.html",
            "https://amd.github.io/ama-sdk/v1.1.2/card_management.html",
            "https://amd.github.io/ama-sdk/v1.1.2/encoder_comp_matrix.html",
            "https://ffmpeg.org/ffmpeg.html",
            "https://ffmpeg.org/ffmpeg-resampler.html",
            "https://ffmpeg.org/ffmpeg-devices.html",
            "https://ffmpeg.org/ffmpeg-all.html",
            "https://trac.ffmpeg.org/wiki/Encode/H.264",
            "https://trac.ffmpeg.org/wiki/Encode/H.265",
            "https://trac.ffmpeg.org/wiki/Encode/AV1",
            "https://trac.ffmpeg.org/wiki/Scaling",
            "https://trac.ffmpeg.org/wiki/Null",
            "https://trac.ffmpeg.org/wiki/FilteringGuide",
            ]
        
        )

        collection_name = base_name
        collection = chroma_client.list_collections()
        if collection_name in collection:
            chroma_client.delete_collection(collection_name)
            chroma_client.clear_system_cache()
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(docstore=documents, vector_store=vector_store)

        # Initialize the parser 
        parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20) 
        nodes = parser.get_nodes_from_documents(documents)

        # %pip install ipywidgets
        # index = VectorStoreIndex.from_documents(documents,storage_context=storage_context,show_progress=True)
        index = VectorStoreIndex(nodes,embed_model=embedding_model,storage_context=storage_context,show_progress=True)

    def load_data(self, embedding_type):
        # load index from stored vectors
        if (embedding_type == "llama3" or embedding_type == "llama2"):
            print("load llama3\n")
            base_name = "ma35_rag_base"
            embedding_model = OllamaEmbedding(model_name=embedding_type,ollama_additional_kwargs={"mirostat": 0}) #base_url="http://localhost:11434"
        elif embedding_type == "bce":
            print("load bce\n")
            base_name = "ma35_rag_base_bce"
            embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 256, 'device': 'cuda'}
            embedding_model = HuggingFaceEmbedding(**embed_args)
        else:
            print("embedding model not correct. default llama3\n")
            embedding_model = OllamaEmbedding(model_name="llama3_8b",ollama_additional_kwargs={"mirostat": 0}) #base_url="http://localhost:11434"

        Settings.embed_model = embedding_model

        collection_name = base_name
        collection = chroma_client.list_collections()
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embedding_model,storage_context=storage_context
        )

        return index
    
    def init_query(self,response_mode,reranker_model,template1,template2):
        if response_mode =='simple_summarize':
            query_engine = self.index.as_query_engine(response_mode='simple_summarize', similary_threshold=0.1, similarity_top_k=5)
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": template1}
            )

        if response_mode =='refine':
            query_engine = self.index.as_query_engine(response_mode='refine',similarity_top_k=50, temperature=0.6,node_postprocessors=[reranker_model])
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": template1}
            )
            query_engine.update_prompts(
                {"response_synthesizer:refine_template": template2}
            )
        
        if response_mode =='tree_summarize':
            query_engine = self.index.as_query_engine(response_mode='tree_summarize',similary_threshold=0.1, similarity_top_k=30, node_postprocessors=[reranker_model])
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": template1}
            )
            query_engine.update_prompts(
                {"response_synthesizer:refine_template": template2}
            )
        return query_engine


        
    def chat(self, prompt:str, model: str, temp: float, system:str = "default") -> str:
        query_engine = self.index.as_query_engine(response_mode='simple_summarize', similary_threshold=0.1, similarity_top_k=5)
        options = dict({'temperature' : temp})
        message = {}
        message['role'] = 'user'
        message['content'] = prompt
        self.messages.append(message)
        response = query_engine.query(model=model, messages=self.messages, options=options)
        self.messages.append(response['message'])
        return response['message']['content']
    
    def query(self, question:str)->str:
        message = {}
        message['content'] = question
        self.messages.append(message)
        response = self.query_engine.query(question)
        return response

    
if __name__ == '__main__':
    client = OllamaClient()
    # index = client.load_data("bce")
    # query_engine = index.as_query_engine(response_mode='simple_summarize', similary_threshold=0.1, similarity_top_k=5)
    
    while True:
        print('You :')
        question=input()
        query_response = client.query_engine.query(question)
        print(f"Answer:{query_response.response}")
        contents = ""
        AiMessage = {}
        # for chunk in query_response:
        #     content = chunk['message']['content']
        #     print(content, end='', flush=True)
        #     contents += content
        AiMessage['role'] = 'assistant'
        AiMessage['content'] = contents
        client.append_history(AiMessage)