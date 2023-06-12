from artifacts.dr import document_retrieval
from artifacts.sr import sentence_retrieval
from artifacts.agg import aggregation
import requests
import json
import os
import sys
import csv
if not 'bert_repo' in sys.path:
    sys.path += ['bert_repo']
from artifacts.bert_api import BertClassifier

def fact_check_pipeline(claims, constituency_parser, dependency_parser, wikipedia, directory, task_type='demo',
                        source_lang='en'):
    w = {}
    try:
        #print('Articles retrieval...')
        document_retrieval(task_type, constituency_parser, dependency_parser, claims, wikipedia, source_lang, directory)
        #print('Sentences retrieval...')
        w = sentence_retrieval(task_type, claims, wikipedia, directory, source_lang=source_lang)
        #print('Classification...')
        if len(w) > 0:
            if constituency_parser is None:
                with open(os.path.join(directory, 'pred_demo.tsv'), "r") as f:
                    reader = csv.reader(f, delimiter="\t", quotechar=None)
                    lines = []
                    for line in reader:
                        lines.append(line)
                bert = BertClassifier('artifacts/checkps')
                res = bert.predict(test_lines=lines)
                with open(os.path.join(directory, 'results_demo.csv'), 'w') as f:
                    for line in res:
                        f.write(line)
            else:
                pass
            aggregation(task_type, claims, w, directory)
    except Exception as e:
            print("Exception in internal main")
            print(e)
    return w
