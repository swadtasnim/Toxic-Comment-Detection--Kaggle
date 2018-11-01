import numpy
import json
import pickle
import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import gensim
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
import re

from sklearn.model_selection import cross_val_score,ShuffleSplit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif,f_classif, RFE

import re

def LOG(transformed_training_features, training_labels, transformed_testing_features, testing_labels):
    LOG_model = LogisticRegression(C=1000)
    LOG_model.fit(transformed_training_features, training_labels)
    LOG_model.fit(transformed_training_features, training_labels)
    prediction = LOG_model.predict(transformed_testing_features)

    print "LOG trained"
    print numpy.mean(prediction == testing_labels)
    return LOG_model



tr=pd.read_csv('train.csv')
ts=pd.read_csv('test.csv')

trr=tr[:3000]
print trr

print trr.shape





#print trr
pp=[]
for t in range(len(trr)):
    text=trr.comment_text[t]
    text=text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"doesn't", "does not ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"hasn't", "has not ", text)
    text = re.sub(r"haven't", "have not ", text)
    text = re.sub(r"didn't", "did not ", text)
    text = re.sub(r"mustn't", "must not ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    pp.append(text)


trr['comment_text'] = pd.Series(pp).astype(str)







targets=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

"""from sklearn.feature_extraction import text

"""
tf_vectorizer = CountVectorizer(max_df=0.3, max_features=10000, stop_words='english')
TR = tf_vectorizer.fit_transform(trr.comment_text)
#TS= tf_vectorizer.transform(ts.comment_text)
#TS = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_topics=30, max_iter=500, random_state=1, learning_offset=90)

LTR= lda.fit_transform(TR)
print LTR.shape
print LTR



#TS=vect.transform(ts.comment_text)

#print  TRR.shape
#testing_features = lda.transform(tf2)

#testing_set = ch2.transform(testing_set)
#TS=vect.transform(ts.comment_text)

print TR.shape
print  TR.toarray()
#print TS
#TRR=TR.toarray()


TRR=np.hstack((TR.toarray(),LTR))
print TRR.shape



LOG_model = LogisticRegression(C=1,solver='lbfgs',max_iter=400)
GNB_model = GaussianNB()
MNB_model = MultinomialNB(alpha=.40, fit_prior=False)
SVC_model = LinearSVC(C=1000, tol=.000000001, max_iter=8000)
SGD_model = SGDClassifier(loss="log", penalty="l2", alpha=.000000001, n_iter=3000)
DT_model = DecisionTreeClassifier()
NC_model = NearestCentroid(metric='euclidean', shrink_threshold=None)
KN_model = KNeighborsClassifier(weights="distance")
MLP_model = MLPClassifier(hidden_layer_sizes=(300,), max_iter=3000, activation="tanh")
classifier=[LOG_model]

#sub = pd.read_csv('sample_submission.csv')
for c in classifier:
    print c,"\n"
    for t in targets:

        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        cv_score = np.mean(cross_val_score(c, TR, trr[t], cv=cv))
        cv_score2 = np.mean(cross_val_score(c, TRR, trr[t], cv=cv))



        print "Target: ",t," count: ",cv_score,", with LDA:  ",cv_score2
        #print "Target: ", t, " count: ", cv_score
        #c.fit(TR, trr[t])
        #jj=c.predict(TR)
        #print "Target: ", t, " ", np.mean(jj==trr[t]), "\n"
        #p=c.predict_proba(TS)[:,1]
        #print p
        #sub[t]=p



vect = TfidfVectorizer(max_features=10000,stop_words='english')
TR=vect.fit_transform(trr.comment_text)
TRR=np.hstack((TR.toarray(),LTR))

for c in classifier:
    print c, "\n"
    for t in targets:
        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        cv_score = np.mean(cross_val_score(c, TR, trr[t], cv=cv))
        cv_score2 = np.mean(cross_val_score(c, TRR, trr[t], cv=cv))

        print "Target: ", t, " tfidf: ", cv_score, ", with LDA:  ", cv_score2


#print sub.head()
#sub.to_csv('sub.csv', index=False)