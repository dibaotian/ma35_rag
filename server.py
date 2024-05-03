import streamlit as st
from prompt import *
import time

st.set_page_config(page_title='MA35 RAG')
# st.title('AMD MA35 expert system')

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'streaming_enabled' not in st.session_state:
    st.session_state['streaming_enabled'] = False

def resetChat():
    st.session_state.messages.clear()
    clearHistory()

# options = ["llama2","llama3"]
options = ["llama3"]
selected_model = " "

# embedding_options = ["llama2","llama3","bce"]
embedding_options = ["llama3","bce"]
selected_embedding_model = " "

response_mode_options = ["simple_summarize", "refine", "tree_summarize"]
selected_response_mode = " "


# Define a function to be called when a select box changes
def on_selectbox_change(selected_model):
    # Call your function here
    # For example:
    print("A select box changed! {}".format(selected_model))
    return selected_model

# side bar
with st.sidebar:

    st.markdown('# MA35 RAG ⚙️', unsafe_allow_html=True)

    # Register the on_change function
    st.session_state.selected_model = "llama3"
    st.session_state.selected_embedding_model = "bce"

    st.markdown('## Select model:')
    selected_model = st.selectbox('Large language Models', index=0, placeholder="Choose an Option", options=options)

    # if selected_model != "":
    #     st.markdown(f"### :rainbow[Selected {selected_model}]")
    
    selected_embedding_model = st.selectbox('Embedding Models', index=1, placeholder="Choose an Option", options=embedding_options)
    # if selected_embedding_model != "":
    #     st.markdown(f"### :rainbow[Selected {selected_embedding_model}]")

    selected_response_mode = st.selectbox('Response mode', index=0, placeholder="Choose an Option", options=response_mode_options)


    # if(st.session_state.selected_embedding_model != selected_embedding_model):
    #     st.session_state.selected_embedding_mode = selected_embedding_model
    #     embedding_model = on_selectbox_change(selected_embedding_model)
    #     load_data(embedding_model)

    # Initialize previous selected model
    if 'prev_selected_model' not in st.session_state:
        st.session_state.prev_selected_model = embedding_options[1]

    # Check if the selected model has changed
    if st.session_state.prev_selected_model != selected_embedding_model:
        st.session_state.prev_selected_model = selected_embedding_model
        load_data(selected_embedding_model)



# main page
with st.chat_message(name='assistant'):
    st.markdown('I am ma35d expert, please ask you question')
for message in st.session_state.messages:
    with st.chat_message(name=message['role']):
        st.markdown(message['content'])
if question := st.chat_input('Enter your questions'):
    st.session_state.messages.append({'role' : 'user', 'content' : question})
    with st.chat_message(name='user'):
        st.markdown(question)
    with st.chat_message(name='assistant'):
        with st.spinner('Processing....'):
            full_response = queryWithModel(question=question, model=selected_model)
            st.session_state.messages.append({'role' : 'assistant', 'content' : full_response})
            if st.session_state['streaming_enabled']:
                message_placeholder = st.empty()
                streaming_response = ""
                # Simulate stream of response with milliseconds delay
                for chunk in full_response.split():
                    streaming_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(streaming_response + "▌", unsafe_allow_html=True)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
            else:
                st.markdown(full_response)
    st.button('Reset Chat 🗑️', use_container_width=True, on_click=resetChat)