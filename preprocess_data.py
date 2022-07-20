import load_data as ld
import json
import numpy as np
import os
import pandas as pd

label_1 = ld.load_data('jsons/Cased Hole')
label_2 = ld.load_data('jsons/LWD')
label_3 = ld.load_data('jsons/Open Hole')


corpus = {}
l = 0


def extract(label, name):
    data = [np.array([])]
    for filename in label:
        if name == '':
            path = 'test/' + filename
        else:
            path = 'jsons/' + name + '/' + filename
        with open(path, 'r') as f:
            distros_dict = json.load(f)
        text = distros_dict['Content']['Curve Information']
        final = np.array([])
        for it in text:
            final = np.append(final, it['name'])
            if it['name'] != 'DEPTH' and it['name'] != 'DEPT':
                corpus[it['name']] = 1
            # l += 1
        data.append(final)
    return data, corpus


def insert_df(data, dt, lab):
    for row in data:
        temp = {}
        if len(row) != 0:
            for attr in dt.columns:
                if attr in row:
                    temp[attr] = 1
                else:
                    temp[attr] = 0
            temp['cat'] = lab
            dt = dt.append(temp, ignore_index=True)
    return dt


def create_df(label_1_data, label_2_data, label_3_data, cols):
    df = pd.DataFrame(columns=cols)
    df = insert_df(label_1_data, df, 1)
    df = insert_df(label_2_data, df, 2)
    df = insert_df(label_3_data, df, 3)
    return df
