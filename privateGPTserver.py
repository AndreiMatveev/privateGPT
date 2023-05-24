#!/usr/bin/env python3

import sys

from dotenv import load_dotenv
from queue import Queue
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from flask import Flask, render_template, request, Response
from streamingcallback import StreamingCallbackHandler
from threading import Thread

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
server = Flask(__name__)

@server.route("/")
def home():
    return render_template("index.html")

@server.route("/get")
def get_gpt_response():

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    queue = Queue()
    callbackhandler = StreamingCallbackHandler(queue)
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

    query = request.args.get('msg')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    def run_qa(qa, query, queue):
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        queue.put(('answer', answer))
        for document in docs:
            queue.put(('doc', document.metadata['source'] + ": " + document.page_content))
        queue.put(('done', None))

#    @stream_with_context
    def generate():
        while True:
            item_type, item = queue.get()
            if item_type == 'answer':
                yield item + '\n'
            elif item_type == 'doc':
                yield item + '\n'
            elif item_type == 'token':
                continue
            elif item_type == 'done':
                break

    Thread(target=run_qa, args=(qa, query, queue)).start()
    responce = Response(generate(), mimetype='text/plain')
    return responce

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=54321, threaded=True)