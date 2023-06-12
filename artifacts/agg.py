import json
import logging
import numpy as np
import pandas as pd
import pickle
import os
import requests
from catboost import CatBoostClassifier, Pool


data_path = os.path.dirname(os.path.abspath(''))
answers = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']    # used to replace numbers and corresponding values
w = {}


def get_prediction_format(claims, pred, bert_res, k=20):
    comb = pred.join(bert_res)
    comb['SUPPORTS'] = comb['SUPPORTS'].fillna(0.0)
    comb['REFUTES'] = comb['REFUTES'].fillna(0.0)
    comb['NOT ENOUGH INFO'] = comb['NOT ENOUGH INFO'].fillna(1.0)

    predictions = []
    for i in range(len(claims)):
        prediction = list(comb.loc[comb['index'] == i][answers].values.flatten())
        while len(prediction) < k * 3:
            prediction += [0.0, 0.0, 1.0]  # padding with NEI prediction
        predictions.append(prediction[:k * 3])
    predictions = np.array(predictions)
    return predictions


def catboost_aggregation(predictions):
    cb_model = CatBoostClassifier()
    cb_model.load_model('artifacts/checkps/catboost_model_inf.dump')
    test_pool = Pool(np.array(predictions))
    probabilities = cb_model.predict_proba(test_pool)
    proba = probabilities.max(axis=1)
    preds = probabilities.argmax(axis=1).flatten().astype(dtype=int).astype(dtype=object)
    preds[preds == 0] = 'SUPPORTS'
    preds[preds == 1] = 'REFUTES'
    preds[preds == 2] = 'NOT ENOUGH INFO'
    res = {"predicted": list(preds), "proba": list(proba)}
    return np.array(res['predicted']), np.array(res['proba'])


def logical_aggregation(predictions, k=20):
    preds = []
    for prediction in predictions:
        scores = []
        ans = []
        for i in range(k):
            ans.append(answers[prediction[i * 3: (i + 1) * 3].argmax()])
            scores.append(np.max(prediction[i * 3: (i + 1) * 3]))
        s = ans.count('SUPPORTS')
        r = ans.count('REFUTES')
        if s == 0 and r == 0:
            preds.append('NOT ENOUGH INFO')
        elif s == 0:
            preds.append('REFUTES')
        elif r == 0:
            preds.append('SUPPORTS')
        else:
            l = ''
            cur_max = 0
            for i in range(len(ans)):
                if ans[i] != 'NOT ENOUGH INFO' and abs(scores[i]) > cur_max:
                    l = ans[i]
                    cur_max = abs(scores[i])
            preds.append(l)
    return np.array(preds), None


def get_sents(label, bert, pred):
    if label == 'NOT ENOUGH INFO':
        return []
    sents = set()
    for i, r in bert.iterrows():
        res = np.array([r['SUPPORTS'], r['REFUTES'], r['NOT ENOUGH INFO']])
        title = pred.loc[i]['title']
        sent = pred.loc[i]['sent']
        text_sent = pred.loc[i]['text_a']
        if answers[np.argmax(res)] == label and len(sents) < 5:
            sents.add(tuple([title, sent, text_sent, answers[np.argmax(res)]]))
    return list(sents)


def get_sents_with_labels(preds, results, bert_res, pred, probas):
    total = len(results)
    predict_sents = []
    predict_labels = []
    if probas is None:
        predict_probas = None
    else:
        predict_probas = []
    start = 0
    for i in range(total):
        end = start
        result = results[i]
        label = preds[i]
        if result == {}:
            predict_labels.append(answers[2])
            predict_sents.append([])
            if predict_probas is not None:
                predict_probas.append(1.)
        else:
            while end < len(pred) and pred.iloc[end]['index'] == pred.iloc[start]['index']:  # search of the relevant part
                end += 1
            bert = bert_res.iloc[start: end - 1]
            start = end
            sents = get_sents(label, bert, pred)
            predict_labels.append(label)
            predict_sents.append(sents)
            if predict_probas is not None:
                predict_probas.append(probas[i])
    return predict_labels, predict_sents, predict_probas


def fever_output_format(predict_labels, predict_sents, predict_probas, ids, out_path='tmp/predictions.jsonl'):
    total = len(predict_labels)
    with open(out_path, 'w', encoding='utf-8') as outfile:
        for j in range(total):
            id = int(ids[j])
            predicted_label = predict_labels[j]
            pred_evidence = predict_sents[j]
            
            predicted_probability = None
            if predict_probas is not None:
                predicted_probability = predict_probas[j]
            
            predicted_evidence = []

            if predicted_label == 'NOT ENOUGH INFO':
                title = None
                sent = None
            else:
                for pred_p in pred_evidence:
                    if pred_p[0] in w:
                        title = w[pred_p[0]]
                        title = pred_p[0]
                        sent = int(pred_p[1])
                        predicted_evidence.append([title, sent])

            data = {
                "id": id,
                "predicted_label": predicted_label,
                "predicted_evidence": predicted_evidence,
                "predicted_probability": predicted_probability
            }
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')


def aggregation(task_type, claims=None, wiki=None, directory='tmp/'):
    global w
    w = wiki
    assert task_type in ['test', 'demo'], "the 'task_type' parameter must take one of two values: 'test' or 'demo'"
    assert task_type == 'test' or claims is not None, "you must provide claims list for 'demo' task type"

    bert_res = pd.read_csv(os.path.join(directory, 'results_{}.csv'.format(task_type)), header=None).rename(
        columns={0: "NOT ENOUGH INFO", 1: "SUPPORTS", 2: "REFUTES"})
    pred = pd.read_csv(os.path.join(directory, 'pred_{}.tsv'.format(task_type)), sep='\t')

    with open(os.path.join(directory, 'results_{}.pickle'.format(task_type)), 'rb') as f:
        results = pickle.load(f)

    if task_type == 'test':
        train = pd.read_csv(data_path + "/FEVER_data/shared_task_test.csv")
        claims = train.claim.values
        ids = train.id.values
    else:
        ids = np.arange(len(claims))

    predictions = get_prediction_format(claims, pred, bert_res)
    preds, probas = catboost_aggregation(predictions)
    predict_labels, predict_sents, predict_probas = get_sents_with_labels(preds, results, bert_res, pred, probas)
    fever_output_format(predict_labels, predict_sents, predict_probas, ids, os.path.join(directory, 'predictions.jsonl'))
    return predictions
