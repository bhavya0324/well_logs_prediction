import preprocess_data as prep
import load_data as ld
import pandas as pd
import pickle
import train as t

data = ld.load_data('test')
test_data, c = prep.extract(data, '')

if len(c) != len(t.corpus):
    print("new attribute encountered, retraining required!")

test_df = pd.DataFrame(columns=t.train_df.columns)

for row in test_data:
    temp = {}
    if len(row) != 0:
        for attr in test_df.columns:
            if attr in row:
                temp[attr] = 1
            else:
                temp[attr] = 0
        test_df = test_df.append(temp, ignore_index=True)

X = test_df.iloc[:, 0:len(test_df.columns) - 1].values

X = t.sc.transform(X)
X_pca = t.pca.transform(X)

pickled_model = pickle.load(open('model.pkl', 'rb'))
y = pickled_model.predict(X_pca)
print(y)
