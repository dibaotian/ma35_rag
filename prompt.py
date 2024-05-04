import streamlit as st
from rag_client import OllamaClient
from typing import List


def returnSystemText(pcap_data : str) -> str:
    PACKET_WHISPERER = f"""
        {st.session_state['system_message']}
        packet_capture_info : {pcap_data}
    """
    return PACKET_WHISPERER

oClient = OllamaClient()

def initLLM(llm) -> None:
    oClient.init_ollama_llm(llm)

def initembed(embeded_model):
    oClient.init_embeded_moedel(embeded_model)

def load_data(embeded_model):
    oClient.load_data(embeded_model)

def init_query(response_mode,reranker_model,template1,template2):
    oClient.init_query(response_mode,reranker_model,template1,template2)

def set_prompt(response_mode):
    template1,template2 = oClient.set_prompt(response_mode)
    return template1,template2

def rank_model_init():
    rank_model = oClient.rank_model_init()
    return rank_model

    

# def initLLM(pcap_data) -> None:
#     oClient.set_system_message(system_message=returnSystemText(pcap_data)) 

# @st.cache_resource
# def getModelList() -> List[str]:
#     return oClient.getModelList()

def chatWithModel(prompt:str, model: str):
    return oClient.chat(prompt=prompt, model=model, temp=0.4)


def queryWithModel(question:str, model: str):
    return oClient.query(question=question)

def clearHistory():
    oClient.clear_history()

def modifySM(new_sm: str) -> None:
    oClient.edit_system_message(new_sm)