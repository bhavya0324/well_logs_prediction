import pickle
import preprocess_data as prep
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import load_data as ld
import numpy as np
from scipy.spatial import ConvexHull

label_1 = ld.load_data('jsons/Cased Hole')
label_2 = ld.load_data('jsons/LWD')
label_3 = ld.load_data('jsons/Open Hole')

print(len(label_1), len(label_2), len(label_3))

label_1_data, corpus = prep.extract(label_1, 'Cased Hole')
label_2_data, corpus = prep.extract(label_2, 'LWD')
label_3_data, corpus = prep.extract(label_3, 'Open Hole')

cols = np.array([])
for ele in corpus:
    cols = np.append(cols, ele)
cols = np.append(cols, 'cat')

train_df = prep.create_df(label_1_data, label_2_data, label_3_data, cols)

dataset = shuffle(train_df, random_state=27)

X = dataset.iloc[:, 0:len(dataset.columns) - 1].values
y = dataset.iloc[:, len(dataset.columns) - 1].values
y = y.astype('int')

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=101)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
explained_variance = pca.explained_variance_ratio_


plot = sns.scatterplot(X_train_pca[:, 0], X_train_pca[:, 1], hue=y_train, palette=['blue', 'green', 'red'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("First two principal components after scaling")
plt.show()

Xax = X_train_pca[:, 0]
Yax = X_train_pca[:, 1]
Zax = X_train_pca[:, 2]

cdict = {1: 'red', 2: 'green', 3: 'blue'}
labl = {1: 'Cased Hole', 2: 'LWD', 3: 'Open Hole'}
marker = {1: 's', 2: 'D', 3: 'o'}
alpha = {1: .5, 2: .4, 3: .3}

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')
for l in np.unique(y_train):
    ix = np.where(y_train == l)
    ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40, label=labl[l], marker=marker[l], alpha=alpha[l])
ax.set_xlabel("First Principal Component", fontsize=9)
ax.set_ylabel("Second Principal Component", fontsize=9)
ax.set_zlabel("Third Principal Component", fontsize=9)

ax.legend()
plt.show()

classifier = LogisticRegression(random_state=27)
classifier.fit(X_train_pca, y_train)
y_pred = classifier.predict(X_val_pca)
cm = confusion_matrix(y_val, y_pred)
print(cm)
print("LR model accuracy:", metrics.accuracy_score(y_val, y_pred))

adb = AdaBoostClassifier()
adb.fit(X_train_pca, y_train)
y_pred = adb.predict(X_val_pca)
cm = confusion_matrix(y_val, y_pred)
print(cm)
print("ADB model accuracy:", metrics.accuracy_score(y_val, y_pred))
pickle.dump(adb, open('model.pkl', 'wb'))
