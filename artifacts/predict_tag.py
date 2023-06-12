import requests
from artifacts.fact_check import fact_check_pipeline
import json
import os
import shutil
import pandas as pd

import random
import pickle
import altair as alt

from langdetect import detect
from iso639 import languages
import wikipedia

import json


def predict_tag(data_input_path, data_output_path):
    sess_id = random.randint(0, 10000000)
    directory = 'tmp_' + str(sess_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        raise Exception('FileExistsError')

    constituency_parser = None  # load_const_parser()
    dependency_parser = None  # load_dep_parser()

    f = open(data_input_path)
    inputdata = json.load(f)
    user_lang = inputdata['language']
    if user_lang not in ['en', 'ru', 'bg', 'fr', 'de', 'it']:
        raise Exception('Unsupported language: {}\n'.format(user_lang))

    input_text = inputdata['sentences']
    claims = [claim.strip() for claim in input_text.split('\n')[:5] if len(claim.strip()) != 0]

    lang = None
    good_lang = True
    bad_length = []
    for i, claim in enumerate(claims):
        lang_c = detect(claim)
        if lang is not None and lang_c != lang:
            good_lang = False
        lang = lang_c
        if len(claim.split()) <= 3 or len(claim) < 15:
            bad_length.append(claims[i])
    if user_lang is not None:
        lang = user_lang

    if len(bad_length) != 0:
        raise Exception('Some claims are too short: {}\n'.format(' ; '.join(bad_length)))
    elif not good_lang:
        raise Exception('Input claims probably have different languages\n')
    else:
        wikipedia.set_lang(lang)
        w = fact_check_pipeline(claims, constituency_parser, dependency_parser, wikipedia, directory, source_lang=lang)
        res_dict = []
        if len(w) == 0:
            raise Exception('Not enough information')
        else:
            with open(os.path.join(directory, 'queries_demo.pickle'), 'rb') as f:
                queries = pickle.load(f)
            with open(os.path.join(directory, 'w_links.pickle'), 'rb') as f:
                w_links = pickle.load(f)
            prediction_table = pd.concat([pd.read_csv(os.path.join(directory, 'pred_demo.tsv'), sep='\t'),
                               pd.read_csv(os.path.join(directory, 'results_demo.csv'), header=None).rename(
                                   columns={0: "NOT ENOUGH INFO", 1: "SUPPORTS", 2: "REFUTES"})],
                              axis=1)
            with open(os.path.join(directory, 'predictions.jsonl'), 'r') as f:
                for i, line in enumerate(f):
                    res = {}
                    claiminfo = []
                    theclaim = {}
                    res = json.loads(line)
                    table_part = prediction_table[prediction_table['index'] == i].sort_values(by=['NOT ENOUGH INFO'])
                    if res['predicted_label'] == 'SUPPORTS':
                        claiminfo.append('True')
                    elif res['predicted_label'] == 'REFUTES':
                        claiminfo.append('False')
                    else:
                        claiminfo.append('Not enough information')

                    if res['predicted_probability'] is not None:
                        claiminfo.append('Confidence %: {}'.format(int(100 * res['predicted_probability'])))
                    else:
                        if res['predicted_label'] == 'NOT ENOUGH INFO':
                            claiminfo.append('Confidence %: {}'.format(int(100 * table_part['NOT ENOUGH INFO'].min())))
                        else:
                            table_part['prediction'] = table_part[['NOT ENOUGH INFO', 'SUPPORTS', 'REFUTES']].idxmax(
                                axis=1).values
                            tr, contr = 'SUPPORTS', 'REFUTES'
                            if res['predicted_label'] == 'REFUTES':
                                tr, contr = 'REFUTES', 'SUPPORTS'
                            score = 1
                            if (table_part['prediction'] == contr).sum() != 0:
                                score = (table_part['prediction'] == tr).sum() / (table_part['prediction'] == contr).sum()
                            claiminfo.append('Confidence %: {}'.format(int(100 * score * table_part[res['predicted_label']].max())))

                    order = []
                    data = dict()
                    supports = {}
                    refutes = {}
                    nei = {}
                    total_sents = {}
                    for j, row in table_part.iterrows():
                        if row['title'] not in order:
                            order.append(row['title'])
                    for title in order:
                        data.setdefault(title, [])
                        supports.setdefault(title, [])
                        refutes.setdefault(title, [])
                        nei.setdefault(title, [])
                        for j, row in table_part[table_part['title'] == title].sort_values(by=['sent']).iterrows():
                            text = w[title]
                            sentence = text[int(row['sent'])]
                            if len(sentence.split()) > 3:
                                data[title].append([sentence, int(row['sent']) + 1])
                                supports[title].append(row["SUPPORTS"])
                                refutes[title].append(row["REFUTES"])
                                nei[title].append(row["NOT ENOUGH INFO"])
                                total_sents[title] = len(text)

                    claiminfo.append('Retrieved articles:')
                    for title in data:
                        claiminfo.append('[{}]({})'.format(title, w_links[title]))
                        claiminfo.append('Retrieved by the phrase: {}'.format(queries[i][title]))

                        probs_array = sum(list(map(list, zip(supports[title], refutes[title], nei[title]))), [])
                        labels_array = ['SUP', 'REF', 'NEI'] * len(data[title])
                        #colors_array = ['green', 'red', 'orange'] * len(data[title])
                        line_numbers = sum([[num, num, num] for _, num in data[title]], [])
                        sentences = sum([[sentence, sentence, sentence] for sentence, _ in data[title]], [])
                        source = {
                            'probability': probs_array,
                            'label': labels_array,
                            'c': [i for i in range(len(probs_array))],
                            'line number': line_numbers,
                            'sentence': sentences
                        }
                        sourcepd = pd.DataFrame({
                            'probability': probs_array,
                            'label': labels_array,
                            'c': [i for i in range(len(probs_array))],
                            'line number': line_numbers,
                            'sentence': sentences
                        })
                        claiminfo.append(source)
                        #res_dict[claims[res['id']]] = claiminfo
                        c = alt.Chart(sourcepd).mark_bar().encode(
                            x=alt.X('c:O', axis=alt.Axis(labels=False, ticks=False, title=" ")),
                            y="probability",
                            color=alt.Color('label',
                                 scale=alt.Scale(
                                 domain=['SUP', 'REF', 'NEI'],
                                 range=['mediumseagreen', 'indianred', 'sandybrown'])),
                            tooltip=['line number', 'probability', 'label', 'sentence']
                        )
                        c.save(f"{data_output_path}/chart_{claims[res['id']]}_.json")
                        theclaim['sentence:'] = claims[res['id']]
                        theclaim['data'] = claiminfo
                        res_dict.append(theclaim)
        json_object = json.dumps(res_dict)
        completename = os.path.join(data_output_path, "results.json")
        with open(completename, "w") as outfile:
            outfile.write(json_object)
        print(res_dict)
