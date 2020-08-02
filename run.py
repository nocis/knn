from sklearn.datasets import load_iris
from sklearn import preprocessing as pps
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import pandas as pd
import seaborn as sns
from knnClassifier import evaluateKnn
from knnClassifier import knnClassifier
from knnClassifier import accuracy
import time

def encodeNoneOrdinalCategory(data, attr):
    dummies = pd.get_dummies(data[attr]).rename(columns=lambda x: attr+'_'+str(x))
    data = pd.concat([data, dummies], axis=1)
    data = data.drop(attr, axis=1)
    return data

def readBank(path, processedPath):
    data = pd.read_csv(path, sep=';')
    data = shuffle(data)

    #impure
    data = data[data['y'] != 'unknown'] 

    #bin the age
    data['age'] = pd.qcut(data['age'], 10)
    data['age'] = pd.factorize(data['age'])[0]+1

    #encode job
    data = encodeNoneOrdinalCategory(data, 'job')

    #encode marital
    data = encodeNoneOrdinalCategory(data, 'marital')

    #encode education
    education = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",  "professional.course", "university.degree"]
    for i in range(len(education)):
        data.loc[data['education'] == education[i], 'education'] = i

    #encode default
    data.loc[data['default'] == 'no', 'default'] = 0
    data.loc[data['default'] == 'yes', 'default'] = 1

    #encode housing
    data.loc[data['housing'] == 'no', 'housing'] = 0
    data.loc[data['housing'] == 'yes', 'housing'] = 1

    #encode loan
    data.loc[data['loan'] == 'no', 'loan'] = 0
    data.loc[data['loan'] == 'yes', 'loan'] = 1

    #encode contact
    data = encodeNoneOrdinalCategory(data, 'contact')

    #encode month
    data = encodeNoneOrdinalCategory(data, 'month')

    #encode day_of_week
    data = encodeNoneOrdinalCategory(data, 'day_of_week')

    #encode poutcome
    data = encodeNoneOrdinalCategory(data, 'poutcome')

    #encode y
    data.loc[data['y'] == 'no', 'y'] = 0
    data.loc[data['y'] == 'yes', 'y'] = 1
    dummies = data.loc[:, 'y']
    data = data.drop('y', axis=1)
    data = pd.concat([data, dummies], axis=1)

    #impure by mean
    data.replace('unknown', np.nan, inplace = True)
    for column in list(data.columns[data.isnull().sum() > 0]):
        mean_val = data[column].mean()
        data[column].fillna(mean_val, inplace=True)

    #scale durarion
    data['duration'] = pps.StandardScaler().fit_transform(data['duration'].values.reshape(-1,1))
    #for realistic prediction
    data = data.drop('duration', axis=1)
    #sacle campaign
    data['campaign'] = pps.StandardScaler().fit_transform(data['campaign'].values.reshape(-1,1))
    #sacle pdays
    data['pdays'] = pps.StandardScaler().fit_transform(data['pdays'].values.reshape(-1,1))
    #sacle previous
    data['previous'] = pps.StandardScaler().fit_transform(data['previous'].values.reshape(-1,1))
    #sacle emp.var.rate
    data['emp.var.rate'] = pps.StandardScaler().fit_transform(data['emp.var.rate'].values.reshape(-1,1))
    #sacle cons.price.idx
    data['cons.price.idx'] = pps.StandardScaler().fit_transform(data['cons.price.idx'].values.reshape(-1,1))
    #sacle cons.conf.idx
    data['cons.conf.idx'] = pps.StandardScaler().fit_transform(data['cons.conf.idx'].values.reshape(-1,1))
    #sacle euribor3m
    data['euribor3m'] = pps.StandardScaler().fit_transform(data['euribor3m'].values.reshape(-1,1))
    #sacle nr.employed
    data['nr.employed'] = pps.StandardScaler().fit_transform(data['nr.employed'].values.reshape(-1,1))

    #sns.set(rc = {'figure.figsize' : (15, 12)})
    #corr = data.corr()
    #corr_map = sns.heatmap(corr, annot = True, fmt = ".1g", cmap = "coolwarm")
    #corr_map.get_figure().savefig('heatmap.jpg')
    data.to_csv(processedPath, index=False)

def doPCA(train ,n):
    pca = PCA(n_components=n)
    reduction = list(pca.fit_transform(train))
    for i in range(len(reduction)):
        reduction[i][-1] =  train.iat[i, -1]
    return reduction

def downSample(data):
    train = data
    Pnum = train[train['y']==1]['y'].count()
    Nnum = train[train['y']==0]['y'].count()

    cutNum = Nnum - Pnum
    for _ in range(cutNum):
        idx = random.randint(0,len(train)-1)
        while train.iat[idx, -1] == 1:
            idx = random.randint(0,len(train)-1)
        train.drop(train.index[idx], inplace=True)

    train = shuffle(train)
    return train

def overSample(data):
    train = data
    Pnum = train[train['y']==1]['y'].count()
    Nnum = train[train['y']==0]['y'].count()

    cutNum = Nnum - Pnum
    Pdata = train[train['y'] == 1] 
    while cutNum - Pnum > 0:
        train = pd.concat([train, Pdata])
        cutNum -= Pnum

    train = shuffle(train)
    return train

def trainKnn(data, nknum):
    nk = np.arange(1, nknum)
    test_accuracy = []
    for k in range(nknum-1):
        scores = evaluateKnn(data, 10, k+1, True)
        print(k)
        #print(np.array(scores).mean())
        test_accuracy.append(np.array(scores).mean())

    plt.figure(figsize=[13,8])
    plt.plot(nk, test_accuracy, label = 'Accuracy')
    plt.legend()
    plt.title('-value VS Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(nk)
    plt.savefig('graph.png')
    bestK = 1+test_accuracy.index(np.max(test_accuracy))
    print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),bestK))
    return bestK

def FeatureEngineering(data, VarHold,PCADIM):
    train = data
    #feature selection, cannot after oversample, data repeat cause ValueError
    y = train.loc[:, 'y']
    x = train.drop('y', axis=1)   
    xSelected = VarianceThreshold(threshold=VarHold).fit_transform(x)
    #print(xSelected.shape)
    xSelected = pd.DataFrame(xSelected)
    train = pd.concat([xSelected, y], axis=1)

    #print(train[train['y']==1]['y'].count())
    #positive oversample or negative downsample
    #train = downSample(train)
    train = overSample(train)
    print(train[train['y']==1]['y'].count())
    print(train[train['y']==0]['y'].count())

    #feature reduction
    reduction = doPCA(train, PCADIM)
    return reduction

def testKnn(data, ratio, *args):
    data = pd.DataFrame(data)
    data = data.sample(frac=1.0)
    data = data.reset_index()
    trainSplit = data.sample(frac=ratio)
    testSplit = data[~data.index.isin(trainSplit.index)]
    trainSplit = trainSplit.to_numpy()
    testSplit = testSplit.to_numpy()
    true = testSplit
    testSplit = []
    #print(trainSplit.shape)
    for row in true:
        rowCopy = list(row)
        rowCopy[-1] = 0
        testSplit.append(rowCopy)

    (pred, probs) = knnClassifier(trainSplit, np.array(testSplit), *args)
    yTrue = [row[-1] for row in true]
    yPred = [row[-1] for row in pred]
    yScore = [row[-1] for row in probs]
    
    acc = accuracy(yTrue, yPred)
    confuseM = confusion_matrix(yTrue, yPred)
    plt.cla()
    plt.matshow(confuseM, cmap='gray_r')
    plt.colorbar()
    for x in range(len(confuseM)):
        for y in range(len(confuseM)):
            plt.annotate(confuseM[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')

    plt.cla()
    precision, recall, _ = precision_recall_curve(yTrue, yScore)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve Acc={0:0.2f}'.format(acc))
    plt.savefig('precision_recall_curve.png')


    plt.cla()
    fpr, tpr, _ = roc_curve(yTrue, yScore)
    AUC = auc(fpr, tpr)
    plt.step(fpr, tpr, color='r', alpha=0.2, where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='r')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC curve:AUC={0:0.2f}'.format(AUC))
    plt.savefig('roc_curve.png')

#iris = load_iris()
#x = iris.data
#y = iris.target
#train = []
#for i in range(len(x)):
#    t = x[i].tolist()
#    t.append(y[i])
#    train.append(t)



#readBank("bank-additional-full.csv", "bank-additional-full-processed.csv")
#train = pd.read_csv('bank-additional-full-processed.csv')
readBank("bank-additional.csv", "bank-additional-processed.csv")
train = pd.read_csv('bank-additional-processed.csv')
train = FeatureEngineering(train, 0.5, 10)
test = train
#bestK = trainKnn(train, 20)


start=time.time()
testKnn(test, 0.8, 10, True)
end=time.time()
print('time:',end-start,'s')