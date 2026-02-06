import argparse
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from language_model import get_retriever
from utils import get_qs

parser = argparse.ArgumentParser()
parser.add_argument("llm", help="LLM to be used")
parser.add_argument("benchmark", help="Benchmark to be used")
parser.add_argument("--multithread", action="store_true", help="Enable multithreading")
args = parser.parse_args()

'''
Params
'''
retriever = get_retriever()
csv_file_name = f"evidences_{args.llm}_{args.benchmark}.csv"

'''
Start of Program
'''
#Create a threading lock to ensure safe writing when multithreading
lock = threading.Lock()

# Retrieve evidences for a question and write them to a CSV file
def retrieve_and_write_csv(args):
    qid, question = args

    #Use the retriever to get evidence strings for the question
    evidences = retriever.retrieve_evidences(question)

    evidences_with_ids = []

    #Attach question ID and rank to each piece of evidence
    for evidence_ind, evidence in enumerate(evidences):
        evidences_with_ids.append([qid, evidence_ind + 1, evidence])

    evidences_df = pd.DataFrame(evidences_with_ids)

    #Write results to CSV
    with lock:
        evidences_df.to_csv(csv_file_name, header=None, encoding='utf-8', mode='a', index=False)

#Define the header for the CSV output
header = ['question_id', 'evidence_relevance_rank', 'evidence']

if not os.path.isfile(csv_file_name):
    with open(csv_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([header])

#Load already processed questions from existing CSV
qs = pd.read_csv(csv_file_name)
existing_qs = set()
for q in qs['question_id'].unique():
    existing_qs.add(q)

unanswered_questions = get_qs(existing_qs, False, False)

if len(unanswered_questions.keys()) > 0:
    #Prepare input tuples of (question_id, question_text) for retrieval
    inputs = []
    for k, v in unanswered_questions.items():
        inputs.append((k, v))

    print(f'Number of unanswered questions: {len(inputs)}')

    #Use a thread pool to parallelize processing
    if args.multithread:
        print("Multithreading")
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 5) as executor:
            for _ in tqdm(executor.map(retrieve_and_write_csv, inputs), total=len(inputs)):
                pass
    else:
        for inp in tqdm(inputs, total=len(inputs)):
            retrieve_and_write_csv(inp)
