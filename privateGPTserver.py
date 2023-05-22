#!/usr/bin/env python3

from dotenv import load_dotenv
from collections import deque
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from flask import Flask, render_template, request
from streamingcallback import StreamingCallbackHandler

import argparse
import os


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

args = parse_arguments()
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever()
callbackhandler = StreamingCallbackHandler()
callbacks = [] if args.mute_stream else [callbackhandler]

# Prepare the LLM
match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    case _default:
        print(f"Model {model_type} not supported!")
        exit;

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

server = Flask(__name__)

@server.route("/")
def home():
    return render_template("index.html")

@server.route("/get")
def get_gpt_response():
    query = request.args.get('msg')
    res = qa(query)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']

    response = answer
    for document in docs:
        response += "\n" + document.metadata['source'] + ": " + document.page_content + "\n"

    return response

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=54321)