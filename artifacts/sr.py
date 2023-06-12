import csv
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from unidecode import unidecode
from mediawiki import MediaWiki
import nltk
import time
import re
from deep_translator import GoogleTranslator

import requests
import bs4


data_path = os.path.dirname(os.path.abspath(''))
wiki_path = os.path.join(data_path, "FEVER_data/wiki_pages.csv")
glove_path = os.path.join(data_path, 'glove.840B.300d.txt')
fasttext_path = os.path.join(data_path, 'cc.en.300.vec')
infersent_path = os.path.join(data_path, "encoder/infersent%s.pkl")

w = {}
wiki = None    # MediaWiki()    # user_agent='WhatTheWikiFact-pymediawiki'
w2v_embs = {}
vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, max_df=0.85, binary=True)


def get_wiki_text(url, source_lang):
    res = requests.get('https://{}.wikipedia.org/wiki/'.format(source_lang) + url)
    res.raise_for_status()
    res = bs4.BeautifulSoup(res.text, 'html.parser')
    t = []
    for i in res.select('p'):
        t.append(re.sub(r'\[\d+\]', '', i.getText().replace('\xa0', ' ')))
    return ''.join(t)


def process_wiki_text(text, src_lang='en'):
    parsed_text = []
    for elem in text.split('\n'):
        if elem == '== See also ==':
            break
        if len(elem) > 0 and elem[0] != '=':
            parsed_text.append(elem)
    parsed_text = ' '.join(parsed_text)
    if src_lang == 'ru':
        return nltk.sent_tokenize(parsed_text, language='russian')
    return nltk.sent_tokenize(parsed_text)


def process_claim_text(claim):
    claim = re.sub(r"\S*https?:\S*", "", claim)
    return ' '.join([word for word in nltk.word_tokenize(claim) if word not in string.punctuation])


def translate(text, source_lang):
    if source_lang == 'en':
        return text
    return GoogleTranslator(source=source_lang, target='en').translate(text)


def unidecode_text(text, source_lang):
    if source_lang == 'en':
        return unidecode(text)
    return text


def w2v(word):
    return w2v_embs.get(word, np.zeros(300))


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def find_sentences_in_document(mode, data, claim, title, claim_emb=None, model=None, word_vectors=None,
                               claim_entities=None, argsort=False):
    if data != []:
        sents = coref_in_start_sents(data, title)
        text = coref_in_start(title, sents)
    else:
        if title not in w:
            return [0]
        text = w[title]

    if mode == 1:
        tfidf = vectorizer.fit_transform([claim] + text)
        sims = cosine_similarity(tfidf[0], tfidf).flatten()[1:]

    if argsort:
        return sims.argsort()
    else:
        return sims


def find_sentences(urls, coref, claim, mode, top, model=None, word_vectors=None):
    result = {}
    claim_emb = None

    sims = np.array([])
    lens = [0]
    titles = []

    for title in urls:
        try:
            data = coref.get(title, [])
            similarities = find_sentences_in_document(mode, data, claim, title, claim_emb, model, word_vectors)
            sims = np.concatenate([sims, similarities])
            lens.append(len(similarities) + lens[-1])
            titles.append(title)
        except:
            continue

    lens = np.array(lens)
    for ind in sims.argsort()[-top:]:
        res = lens[lens - ind <= 0].argmax()
        result.setdefault(titles[res], dict())
        result[titles[res]][ind - lens[res]] = sims[ind]

    return result


def save_sr_results(claims, documents, mode, task_type, total=None, top=20, use_coreference=False, ann_type='Stanford', 
                    directory='tmp/'):
    if total is None:
        total = len(claims)

    if use_coreference:
        corefs = read_corefs(ann_type)
    coref = {}

    if w2v_embs == {}:
        if mode == 3 or mode == 5:
            load_glove(glove_path)
        if mode == 4:
            load_fasttext(fasttext_path)

    word_vectors = None
    model = None
    if mode == 6:
        model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
        word_vectors = model.wv

    if mode == 3:
        model = load_infersent(infersent_path)

    saver = []
    for i in range(total):
        urls = []
        claim = claims[i]
        urls = documents[i]
        if use_coreference and i < len(corefs):
            coref = corefs[i]
        result = find_sentences(urls, coref, claim, mode, top, model, word_vectors)
        saver.append(result)

    with open(os.path.join(directory, 'results_{}.pickle'.format(task_type)), 'wb') as f:
        pickle.dump(saver, f)


def create_bert_file(task_type, claims, mode=1, total=None, top=20, use_coreference_bert=False, 
                     use_coreference_search=False, ann_type='Stanford', directory='tmp/', source_lang='en'):
    with open(os.path.join(directory, 'documents_{}.pickle'.format(task_type)), 'rb') as f:
        documents = pickle.load(f)
    w_links = {}
    for el in documents:
        for doc in documents[el]:
            try:
                w[doc] = process_wiki_text(get_wiki_text(doc.replace(' ', '_'), source_lang), source_lang)  # page.content
                w_links[doc] = 'https://{}.wikipedia.org/wiki/'.format(source_lang) + doc.replace(' ', '_')  # page.url
            except:
                pass
    with open(os.path.join(directory, 'w_links.pickle'), 'wb') as f:
        pickle.dump(w_links, f)
    with open(os.path.join(directory, 'w_docs.pickle'), 'wb') as f:
          pickle.dump(w, f)
    save_sr_results(claims, documents, mode, task_type, total, top, use_coreference_search, ann_type, directory)
    with open(os.path.join(directory, 'results_{}.pickle'.format(task_type)), 'rb') as f:
        saver = pickle.load(f)

    if use_coreference_bert:
        corefs = read_corefs(ann_type)
        new_corefs = {}
        for i in range(len(corefs)):
            for title in corefs[i]:
                new_corefs[title] = corefs[i][title]
        del corefs
    
    with open(os.path.join(directory, 'pred_{}.tsv'.format(task_type)), "w", encoding="utf-8") as file:
        f = csv.writer(file, delimiter='\t')
        f.writerow(["text_a", "text_b", "title", "sent", "index"])
        for i in range(len(saver)):
            rows = []
            new_result = {}
            result = saver[i]
            text_b = translate(process_claim_text(claims[i]), source_lang)

            for title in result:
                for sent in result[title]:
                    new_result[(title, sent)] = result[title][sent]

            for el in list(sorted(new_result.items(), key=lambda kv: -kv[1])):
                doc = el[0][0]
                if doc in w:
                    t = doc
                    sent = el[0][1]
                    text = w[doc]

                    if use_coreference_bert:
                        try:
                            data = new_corefs[doc]
                            sents = coref_in_start_sents(data, doc, ann_type)
                            text = coref_in_start(doc, sents)
                        except:
                            pass

                    if text[sent] != '' and text[sent].strip()[0] == '.':
                        text[sent] = text[sent].strip()[1:]
                    if text[sent][-1] == '.':
                        text[sent] = text[sent][:-1]
                    text_a = unidecode_text(t.replace('_', ' '), source_lang) + ' # ' + text[sent] + ' . '
                    rows.append([text_a, text_b, doc, sent, i])

            texts_a = []
            translates = []
            cur_length = 0
            j = 0
            while j < len(rows):
                row = rows[j]
                if cur_length + len(row[0]) < 2000:
                    cur_length += len(row[0])
                    texts_a.extend(row[0].split(' # '))
                    j += 1
                else:
                    if cur_length > 0:
                        translates.extend(translate(' \n '.join(texts_a), source_lang).split('\n'))
                        texts_a = []
                        cur_length = 0
                    else:
                        translates.extend(row[0].split(' # '))
                        j += 1
            if cur_length > 0:
                translates.extend(translate(' \n '.join(texts_a), source_lang).split('\n'))
            assert len(translates) == 2 * len(rows)
            for j, row in enumerate(rows):
                f.writerow(['# ' + translates[2 * j] + ' # ' + translates[2 * j + 1]] + row[1:])


def sentence_retrieval(task_type, claims=None, wikipedia=None, directory='tmp/', mode=1, total=None, top=20,
                       use_coreference_bert=False, use_coreference_search=False, ann_type='Stanford', source_lang='en'):
    assert task_type in ['test', 'eval',
                         'demo'], "the 'task_type' parameter must take one of four values: 'test', 'eval' or 'demo'"
    assert task_type != 'demo' or claims is not None, "claims list should be provided in the 'demo' task type"

    if task_type != 'demo':
        if task_type == 'eval':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
        if task_type == 'test':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_test.csv"))
        claims = train.claim.values
    create_bert_file(task_type, claims, mode, total, top, use_coreference_bert, use_coreference_search, ann_type, directory, source_lang)
    return w
