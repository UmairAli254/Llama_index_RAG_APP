import streamlit as st
import openai
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
# from langchain.embeddings import HuggingFaceEmbeddings


openai.openai_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Llama index", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Llama Index")

if "messages" not in st.session_state:
 st.session_state["messages"] = [
     {
         "role": "assistant",
         "content": "Ask me any Question about your Docs"
     }
 ]


@st.cache_resource(show_spinner=False)
def load_data():
 with st.spinner(text="Loading and Indexing the Document..."):

  docs = SimpleDirectoryReader("data").load_data() #It's loading all document in the data directory

 llm = OpenAI(
     model_name="gpt-3.5-turbo",
     temperature=0.5,
     systemprompt="You are expert on the All Docs uploaded by user, your job is to provide the valid and relevant answers. Keep your answers based on facts do not hallucinate"
 )

 service_content = ServiceContext.from_defaults(
     llm=llm,
     #  The below line/model is to create indexes
     embed_model="local:all-MiniLM-L6-v2",

 )

 # Here am creating the index
 index = VectorStoreIndex.from_documents(
     docs,
     service_context=service_content
 )

 return index


index = load_data()
# st.write(index)
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    verbose=True
)

# Let's Take the Use Input and Store in session
if prompt := st.chat_input("Your Question: "):
 st.session_state.messages.append(
     {
         "role": "user",
         "content": prompt
     }
 )


# display Prior Chat Messages
for message in st.session_state.messages:
 with st.chat_message(message["role"]):
  st.write(message["content"])

# If the last messsage is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
 with st.chat_message("assistant"):
  with st.spinner("Thinking..."):
   res = chat_engine.chat(prompt)
   st.write(res.response)
   message = {
       "role": "assistant",
       "content": res.response
   }
  #  Add Response to message history
   st.session_state.messages.append(message)
