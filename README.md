# Fever for files
This is a redesigned code from the WhatTheWikiFact application (https://arxiv.org/pdf/2105.00826.pdf) - a system for automatic claim verification using Wikipedia. The system can predict the veracity of an input claim, and it further shows the evidence it has retrieved as part of the verification process. It shows confidence scores and a list of relevant Wikipedia articles, together with detailed information about each article, including the phrase used to retrieve it, the most relevant sentences extracted from it and their stance with respect to the input claim, as well as the associated probabilities. 

The overall architecture was preserved.  First, the Document Retrieval (DR) module finds potentially relevant documents from Wikipedia. Then, the Sentence Retrieval (SR) module extracts the top-20 most relevant sentences from these documents. Afterwards, the Natural Language Inference (NLI) module classifies each claim–sentence pair as support/refute/NEI. Finally, the aggregation module makes a final prediction.

## Quick start  
    
### Create virtual enviroment on python 3.7 and move to project directory
    virtualenv myvenv -p /path/to/python3.7
    source myvenv/bin/activate
    cd path/to/directory
### Set up requirements 
    pip install -r reqs3.txt

Create an input json file containing a language and up to 5 claims for verification, then, set up a parameters file with paths to the input data and output data. 

### Finally, run exec_script.py
    python exec_script.py -p /full/path/to/params.json

Your output will be stored in the example_data/output_data folder. It will consist of several json files. The first will contain information about every claim you entered in the format:
### f
   [
{"sentence": "Пушкин родился в...", "data" : "Verdict, Confidence, etc"},
    {"sentence": "Пушкин - певец", "data": "something"}
    ]
Other output files will be Altair charts of the predicted stance labels for the input claim with respect to each retrieved sentence. The stance is expressed as one of the classes Supports (SUP), Refutes (REF), or Not Enough Info (NEI). The chart further shows the class probability, which is also represented as the bar height, sentence number, and label, which is also indicated with the corresponding color. Note that there are three bars for each sentence, i.e., one for each label. Moreover, the bars are ordered (grouped) by sentences.
