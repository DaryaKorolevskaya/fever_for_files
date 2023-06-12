import warnings
warnings.filterwarnings("ignore")

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from allennlp.predictors.predictor import Predictor
import numpy as np
from starlette.responses  import JSONResponse
import os
import pandas as pd
import pickle
from unidecode import unidecode
from mediawiki import MediaWiki
import requests
import json
import re
import stanza
bg_nlp = stanza.Pipeline('bg', processors='tokenize,ner')

from natasha import Doc, Segmenter, NewsEmbedding, NewsNERTagger
from bulstem.stem import BulStemmer

emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
segmenter = Segmenter()

data_path = os.path.dirname(os.path.abspath(''))
ps = PorterStemmer()
ru_stemmer = SnowballStemmer("russian")
fr_nlp = stanza.Pipeline('fr', processors='tokenize,ner')
de_nlp = stanza.Pipeline('de', processors='tokenize,ner')
it_nlp = stanza.Pipeline('it', processors='tokenize,ner')
bg_nlp = stanza.Pipeline('bg', processors='tokenize,ner')
ru_stemmer = SnowballStemmer("russian")
fr_stemmer = SnowballStemmer("french")
de_stemmer = SnowballStemmer("german")
it_stemmer = SnowballStemmer("italian")
bg_stemmer = BulStemmer.from_file('artifacts/stem_rules_context_2_utf8.txt', min_freq=2, left_context=2)

def unidecode_text(text, source_lang):
    if source_lang == 'en':
        return unidecode(text)
    return text


def extract_noun_phrase_from_tree(node):
    if node['nodeType'] == 'NP':
        yield node['word']
    for child in node.get('children', []):
        yield from extract_noun_phrase_from_tree(child)


def extract_titles(search, urls, queries, wiki, stem_claim=None, stem=True, claim=None, source_lang='en'):
    assert not stem or stem_claim is not None

    try:
        for p in wiki.search(search, results=3):  # top-7,5,2 works worse
            title = p.split('(')[0]
            title = re.sub(r"[,.;@#?!&$]+\ *", " ", title)
            if title == 'Trollhunter':
                title = 'Trollhunters'

            in_claim = True  # results filtering
            if source_lang == 'en':
                if stem:
                    stem_title = [unidecode_text(ps.stem(word), source_lang) for word in word_tokenize(title)]
                    for word in stem_title:
                        if word not in stem_claim:
                            in_claim = False
                else:
                    for word in word_tokenize(title):
                        if word not in claim:
                            in_claim = False
            else:
                if source_lang == 'ru':
                    stem_title = [ru_stemmer.stem(word) for word in word_tokenize(title, language='russian')]
                elif source_lang == 'fr':
                    stem_title = [fr_stemmer.stem(word) for word in word_tokenize(title, language='french')]
                elif source_lang == 'de':
                    stem_title = [de_stemmer.stem(word) for word in word_tokenize(title, language='german')]
                elif source_lang == 'it':
                    stem_title = [it_stemmer.stem(word) for word in word_tokenize(title, language='italian')]
                elif source_lang == 'bg':
                    stem_title = [bg_stemmer.stem(word) for word in word_tokenize(title, language='russian')]
                if len(set(stem_title) & set(stem_claim)) == 0:
                    in_claim = False
            if in_claim and 'disambiguation' not in p:
                urls.add(p)
                queries[p] = queries.get(p, search)
    except:
        pass
    return urls, queries


def find_documents(claim, constituency_parser, dependency_parser, wiki, stem=True, use_ner=False, source_lang='en'):
    claim = re.sub(r"[,.;@#?!&$]+\ *", " ", claim)
    if source_lang == 'en':
        tokenized_claim = word_tokenize(claim)
        stem_claim = [unidecode_text(ps.stem(word), source_lang) for word in tokenized_claim]
    elif source_lang == 'ru':
        tokenized_claim = word_tokenize(claim, language='russian')
        stem_claim = [ru_stemmer.stem(word) for word in tokenized_claim]
    elif source_lang == 'fr':
        tokenized_claim = word_tokenize(claim, language='french')
        stem_claim = [fr_stemmer.stem(word) for word in tokenized_claim]
    elif source_lang == 'de':
        tokenized_claim = word_tokenize(claim, language='german')
        stem_claim = [de_stemmer.stem(word) for word in tokenized_claim]
    elif source_lang == 'it':
        tokenized_claim = word_tokenize(claim, language='italian')
        stem_claim = [it_stemmer.stem(word) for word in tokenized_claim]
    else:
        tokenized_claim = word_tokenize(claim, language='russian')
        stem_claim = [bg_stemmer.stem(word) for word in tokenized_claim]
    urls = set()
    queries = {}
    if source_lang == 'en':
        if constituency_parser is None:
            constituency_parser = Predictor.from_path(
        "artifacts/checkps/60c14844468543e4329ce7e8d3444fa1f9f7057b4b0de5b3f4a597eb57113d32.73aa20bab6336a582588814d8458d040b59536ca1f60b6a769a2da61c7aa3c9a")
        res = constituency_parser.predict(sentence=claim)['hierplane_tree']['root']
        for noun_phrase in extract_noun_phrase_from_tree(res):
            urls, queries = extract_titles(noun_phrase, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
    
        if dependency_parser is None:
            dependency_parser = Predictor.from_path(
                'artifacts/checkps/c58a56ec0a80151bfe4aafafb2176f645fe144e7c11e90cf2d328fa01cf6e293.94fce0bb982c6ef65a0383759ddd1e337b9ba8540bf8342730b3df279823431e')

        word = dependency_parser.predict(sentence=claim)['hierplane_tree']['root']['word']
        prev = claim.split(word)[0]
        if prev != '':
            urls, queries = extract_titles(prev, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
        
        if use_ner:
            res = ner_parser.predict(sentence=claim)
            ner = ' '.join(np.array(res['words'])[np.array(res['tags']) != 'O'])
            if ner != '':
                urls, queries = extract_titles(ner, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
    elif source_lang == 'ru':
        doc = Doc(claim)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        for ne in doc.spans:
            urls, queries = extract_titles(ne.text, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
        urls, queries = extract_titles(claim, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
    elif source_lang == 'fr':
        doc = Doc(claim)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        for ne in doc.spans:
            urls, queries = extract_titles(ne.text, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
        urls, queries = extract_titles(claim, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
    elif source_lang == 'de':
        doc = Doc(claim)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        for ne in doc.spans:
            urls, queries = extract_titles(ne.text, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
        urls, queries = extract_titles(claim, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
    elif source_lang == 'it':
        doc = Doc(claim)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        for ne in doc.spans:
            urls, queries = extract_titles(ne.text, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
        urls, queries = extract_titles(claim, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
    elif source_lang == 'bg':
        doc = bg_nlp(claim)
        res = [ne.text for ne in doc.sentences[0].build_ents()]
        for ne in res:
            urls, queries = extract_titles(ne, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)
        urls, queries = extract_titles(claim, urls, queries, wiki, stem_claim, stem, tokenized_claim, source_lang)

    return urls, queries


def document_retrieval(task_type, constituency_parser, dependency_parser, claims=None, wikipedia=None,
                       lang='en', directory='tmp/'):
    assert task_type in ['test', 'train', 'eval',
                         'demo'], "the 'task_type' parameter must take one of four values: 'test', 'train', 'eval' or 'demo'"
    assert task_type != 'demo' or claims is not None, "claims list should be provided in the 'demo' task type"
    
    wiki = wikipedia
    source_lang = lang
    '''
    if task_type != 'demo':
        if task_type == 'train':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/train.csv"))
        if task_type == 'eval':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
        else:
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_test.csv"))
        claims = train.claim.values
        verifiable = train.verifiable.values
        '''
    documents = {}
    queries_all = {}
    length = len(claims)
    for i in range(length):
        if task_type != 'train' or verifiable[i] != 'VERIFIABLE':
            claim = claims[i]
            urls, queries = find_documents(claim, constituency_parser, dependency_parser, wiki, source_lang=source_lang)
            documents[i] = tuple(urls)
            queries_all[i] = queries

    with open(os.path.join(directory, 'documents_{}.pickle'.format(task_type)), 'wb') as f:
        pickle.dump(documents, f)
    with open(os.path.join(directory, 'queries_{}.pickle'.format(task_type)), 'wb') as f:
        pickle.dump(queries_all, f)
