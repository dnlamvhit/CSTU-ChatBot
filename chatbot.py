# app.py
# Team 2 - GenAI project
#!pip install gensim nltk
import gensim
import nltk
nltk.download('punkt')
from gensim.models import Word2Vec

import streamlit as st
import pandas as pd
import random
import time
import openai
import textwrap3 as textwrap
import dotenv
from dotenv import load_dotenv
import numpy as np
import os
from pinecone import Pinecone
import os
import csv

# For sending email
import json
import sendgrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from joblib import load

#embedding_model = load('CSTU-embedding-model.mdl')
embedding_model = "text-embedding-ada-002"

st.title("CSTU GenAI Chatbot by Team 2 💬")
st.sidebar.image("CSTU.png",use_column_width=True)
st.sidebar.markdown("<font color='darkblue'><b>Course: CSE642 - GenAI</b></font>", unsafe_allow_html=True)
st.sidebar.markdown("<font color='darkblue'><b>Professor: Sridharan Muthuswamy</b></font>", unsafe_allow_html=True)
st.sidebar.markdown("<font color='darkblue'><b>Students: Lam Dao & Fang Wang</b></font>", unsafe_allow_html=True)
st.sidebar.markdown("<font color='darkblue'><b>Released time: February 17th, 2024</b></font>", unsafe_allow_html=True)
st.sidebar.image("robo.gif",use_column_width=True)

# Generate an embedding for a text
def generate_embedding(model, text):
    tokens = nltk.word_tokenize(text)
    word_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not word_vectors: # If no valid word vectors are found, return a vector of zeros
        return np.zeros(model.vector_size)
    embedding = np.mean(word_vectors, axis=0)
    return embedding

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    SENDGRID_API_KEY = st.secrets["SENDGRID_API_KEY"]
except Exception as e:
    # Secrets not found in Streamlit, try loading from local .env file
    dotenv_path = 'D:\.env'  # Specify the path to the .env file
    load_dotenv(dotenv_path) 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

    if not OPENAI_API_KEY or not PINECONE_API_KEY or not SENDGRID_API_KEY:
        st.error("Environment file error or secrets not found!")
        st.error(e)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

index_name = 'cstugpt-kb'
pc = Pinecone( # initialize connection to pinecone
    api_key=PINECONE_API_KEY,
    environment="us-west1-gcp-free")
index = pc.Index(index_name) # connect to pinecone index

if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []    

# Initialize chat history
delimiter = ""
if "prompt_history" not in st.session_state: # Initialize the chat history with the system message if it doesn't exist
        st.session_state.prompt_history = [
            {'role': 'system', 'content': f"""\
You will answer questions about California Science and Technology University (CSTU) based on contents provided at system role. At fisrt, welcome to CSTU.\
If users ask to register courses, offer available courses for registration and ask them to select courses. After they finish selection, let summarize selected courses and ask for their name and email. If they provide name and email, complete registration.\
If they want to get registration record, ask their email. If they provide email, call function get_registration.\
If they want to get course grades, ask their email. If they provide email, call function get_grades.\
If user want to update course grades, ask secrete code. If they give a code other than 2024, tell them invalid code and ask them try again. If they enter 2024, call function updade_grades.\
If they inquiry information related to CSTU out of provided context, ask them to check the website www.cstu.edu or email admission@cstu.edu, Tel 408-400-3948."""} ]
"""
"""
# During the coversation, refer to chat history and the information delimited by {delimiter}.
def chat_complete_messages(messages, temperature=0):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        functions = [
         {
            "name": "registration",
            "description": "complete registration",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_name": {"type":"string","description":"The name of the user",},
                    "student_email": {"type": "string", "description": "The email of user",},
                    "courses":{"type":"string", "description":"The courses the user want to register",},
                    "body": {"type": "string", "description": "Confirmation content of CSTU about courses registered by user",},
                },
                "required": ["student_name", "student_email", "courses","body"],
            }
         },
        {
            "name": "get_registration",
            "description": "get registration record",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_email": {"type": "string", "description": "The email of the student user",}
                },
                "required": ["student_email"],
            }
         },
        {
            "name": "update_grades",
            "description": "update course grades",
            "parameters": {
                "type": "object",
                "properties": {
                    "prof_code": {"type": "string", "description": "The secrete code of the professor",}
                },
                #"required": ["prof_code"],
            }
         },
        {
            "name": "get_grades",
            "description": "get course grades",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_email": {"type": "string", "description": "The email of the student user",}
                },
                "required": ["student_email"],
            }
         },
        ],
       function_call="auto",
    )
    return response.choices[0]["message"]

def limit_line_width(text, max_line_width):
    """ Function to limit the line width of the text """
    if text is None: return ""
    lines = textwrap.wrap(text, width=max_line_width)
    return "\n".join(lines)

def registration(student_name,student_email,courses,body):
    try:
        csv_file = "registration_records.csv"
        data = [time.strftime("%Y-%m-%d %H:%M:%S"), student_name, student_email, courses]
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["REGISTRATION TIME","STUDENT NAME", "EMAIL ADDRESS", "COURSE NAME"])
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        message = Mail(
            from_email='cstu02@gmail.com',
            to_emails=student_email,
            subject='Course registration confirmation from CSTU',
            html_content=body)
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
    except Exception as e:
            print(e.message)
            st.info("A registration confirmation message has been sent to your email.")
    
def get_registration(student_email):
    try:
        df = pd.read_csv("registration_records.csv")
        result = df[df["EMAIL ADDRESS"] == student_email].to_dict()
        del df
    except Exception as e:
        result = f"Registration records not found for {student_email}!"
    return result


def update_grades(prof_code):
  new_grades = st.file_uploader("Select your new grades file", type="csv")
  time.sleep(30)
  #while new_grades is None: pass
      #if prof_code==2004:  
  st.write("Grades updated!") 
  try:
            df = pd.read_csv("newgrades.csv")
            print(df)
            if os.path.exists("grades.csv"):
                df.to_csv("grades.csv", mode="a", index=False, header=False)
            else:
                df.to_csv("grades.csv", index=False)
            st.balloons()
            del new_grades 
            return("Grades updated!")
  except Exception as e:
            st.write(e)
      #else: st.write("Your secrete code is invalid")
      
def get_grades(student_email):
    try:
        df = pd.read_csv("grades.csv")
        result = df[df["student_email"] == student_email].to_dict()
        del df
    except Exception as e:
        result = f"Grades records not found for {student_email}"
    return result

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Accept user input
if user_input := st.chat_input("Welcome to CSTU Chatbot of GenAI Team 2! 🤖"):
    if OPENAI_API_KEY:
        # Word2Vector embedding
        # input_emb=generate_embedding(embedding_model, user_input)
        # OpenAI embedding
        res = openai.Embedding.create(input=[user_input],engine=embedding_model)
        input_emb=res['data'][0]['embedding'] 
        kb_res = index.query(vector=input_emb, top_k=1, include_metadata=True, namespace='cstu', metric="cosine")
        #If the include_metadata parameter is set to True, the query method will only return the id, score, and metadata for each document. The vector for each document will not be returned
        metadata_text_list = [x['metadata']['text'] for x in kb_res['matches']]
        limit = 3600  #set the limit of knowledge base words
        kb_content = " "
        count = 0
        proceed = True
        while proceed and count < len(metadata_text_list):  # append until hitting limit
            if len(kb_content) + len(metadata_text_list[count]) >= limit:
                proceed = False
            else:
                    kb_content += metadata_text_list[count]
            count += 1
        knowledge_message = {"role": "system", "content": f"""
                             {delimiter}{kb_content}{delimiter}
                             """}
       
        # Add user message to chat history
        user_message = {"role": "user", "content": user_input}
        st.session_state.chat_history.append(user_message)
        
        # Add knowledge base and user message to promt history      
        st.session_state.prompt_history.append(knowledge_message)        
        st.session_state.prompt_history.append(user_message)

        # Get the model response
        response = chat_complete_messages(st.session_state.prompt_history, temperature=0)
        #response = chat_complete_messages(C, temperature=0)
        # Limit the line width to, for example, 60 characters
        max_line_width = 60

        if response.get("function_call"): # e.g. Sending email
            function_name = response["function_call"]["name"]
            # print("function_name: ",function_name)
            # function_to_call = available_functions[function_name]
            # print("function_to_call: ", function_to_call)
            function_args = json.loads(response["function_call"]["arguments"])
            if function_name == 'registration':
                registration(function_args.get("student_name"), function_args.get("student_email"), function_args.get("courses"), function_args.get("body"))
                formatted_text = "Thank you for providing your email address. A confirmation message for your registration has been sent to your email. Please check it and let me known if there is any further requirement."
                #st.info("The following message has been sent to "+function_args.get("student_email")+":\n"+function_args.get("body"))
            elif function_name == 'get_registration':
                # print(function_args.get("student_email"))
                result = get_registration(function_args.get("student_email")) 
                formatted_text = f"{result}"
            elif function_name == 'get_grades':
                result = get_grades(function_args.get("student_email")) 
                formatted_text = f"{result}"
            elif function_name == 'update_grades':
                result = update_grades(function_args.get("prof_code")) 
                formatted_text = "Grades updated!"
            else: print("function_name: ",function_name)                       
        else: formatted_text = response["content"]
            # formatted_text = limit_line_width(response["content"], max_line_width)            
        ai_message = {"role": "assistant", "content": formatted_text}
        st.session_state.chat_history.append(ai_message)
        st.session_state.prompt_history.append(ai_message)

        # Display message in chat message container
        with st.chat_message("user"):
            st.write(user_message['content'])
        with st.chat_message("assistant"):
            try:
                st.write(pd.DataFrame(result))
            except Exception as e: 
                st.write(ai_message['content'])
    else:
        st.write("!!! Error: You need to enter OPENAI_API_KEY!")
    
