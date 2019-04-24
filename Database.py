import re
import sklearn
import string
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer

import nltk


import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import Adadelta,Adam,RMSprop

from keras.utils import np_utils





class Database:

    #Queue contains a list of lines only for that firm. (Function Excel)

    def __init__(self):
        #print("Initialization of Join")
        self.trainingVector=[];
        self.Table=[];
        self.classVector=[]

    def LastClean(self,word):

        word=word.replace('</a>', '').replace('</e>', '').replace('<e>','').replace('<a>', '');
        regex= re.compile('[$%&*#,@.!():?]')
        word=regex.sub('', word)
        return word

    def write(self,bit):
        OpenFile3= open(r"D:\CS 583\OutputDatamining.txt", 'a')
        if (bit==1):#Full (Cleaned)-->upper or lower?


            for tweet in self.Table:

                for word in tweet.WordList:

                    # print("Printing Word",self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word)))
                    OpenFile3.write(self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word))+";"+"\t")
                OpenFile3.write(str(tweet.Class)+";"+"\n")
        elif(bit==2):#Stemmed
            for tweet in self.Table:
                for word in tweet.stemmedList:
                    #print("Printing Word",self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word)))
                    OpenFile3.write(self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word))+";"+"\t")

                OpenFile3.write(str(tweet.Class)+";"+"\n")

        elif(bit==3):#Spell Checker and Full(Cleaned)
            print("Printing Spellchecker")

        OpenFile3.close()

    def add(self,Tweet):

        self.Table.append(Tweet)
    def getSize(self):
        return len(self.Table)

    def Vectorizeself(self, ngram_range=(1, 2)):
        #print("Inside Vectorizing")
        #Vectorizing
        rare_words = self.get_rare_words(1)
        stopwords=nltk.corpus.stopwords.words('english')
        for i in range(0,len(stopwords)):
            stopwords[i] = stopwords[i].replace("'","")
        #print(stopwords)
        #print(rare_words)
        #quit()
        wordsToIgnore = list(set(stopwords + rare_words))

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word',max_features=10000,tokenizer=lambda x: x,preprocessor=lambda x: x,token_pattern=None,stop_words=wordsToIgnore, ngram_range=ngram_range)

        stemmedTexts=[o.stemmedList for o in self.Table]
        vectorizer.fit(stemmedTexts)

        self.trainingVector = vectorizer.transform(stemmedTexts).todense()
        print(self.trainingVector.shape)
        #print("after transform",self.trainingVector.toarray())

        print(self.trainingVector.shape)
    def get_rare_words(self,threshold):#self=database
        word_list = []
        for tweet in self.Table:
            for word in tweet.stemmedList:
                word_list.append(word)

        counts = Counter(word_list)
        rare_words = []
        for word in word_list:
            if counts[word] <= threshold:
                rare_words.append(word)

        rare_words = sorted(rare_words)
        return rare_words
    def CreateClassVector(self):
        self.classVector = [o.Class for o in self.Table]

    def classification_report_with_accuracy_score(self,y_true, y_pred):

        print (classification_report(y_true, y_pred)) # print classification report
        print(accuracy_score(y_true, y_pred))
        return accuracy_score(y_true, y_pred) # return accuracy score


    def TrainMultinomialNaiveBias(self):

        #X_train, X_valid, y_train, y_valid = train_test_split(self.trainingVector, self.classVector, test_size = 0.1, shuffle = True)

        scoring = ['precision_weighted','precision_macro','recall_weighted' 'recall_macro','f1_weighted']#Not used if we want the same Output as MObashir
        clf = MultinomialNB()

        scores = cross_validate(clf,self.trainingVector , self.classVector, scoring=make_scorer(self.classification_report_with_accuracy_score),
                                cv=10, return_train_score=False)

        #print("Keys",sorted(scores.keys()))

        print("Scores for Naive Bayes.....")
        #print("Recall:",scores['test_recall_macro'],"Precision:",scores['test_precision_macro'])

        print(scores)


        #clf.fit(X_train,y_train)
        #y_pred = clf.predict(X_valid)
        #print('Multinomial Naive Bayes accuracy: %s' % accuracy_score(y_pred, y_valid))

        #print(classification_report(y_valid, y_pred,target_names=['-1.0','0.0','1.0']))


        #print(y_pred)

    def TrainLinearSVM(self):
        #X_train, X_valid, y_train, y_valid = train_test_split(self.trainingVector, self.classVector, test_size = 0.1, shuffle = True)
        scoring = ['precision_macro', 'recall_macro','f1_weighted']
        clf = OneVsOneClassifier(LinearSVC(random_state=0))
        scores = cross_validate(clf,self.trainingVector, self.classVector, scoring=make_scorer(self.classification_report_with_accuracy_score),
                                cv=10, return_train_score=False)

        print("Scores for LinearSVM.....")
        print(scores)
        #clf.fit(X_train,y_train)
        #y_pred = clf.predict(X_valid)
        #print('Linear SVM accuracy: %s' % accuracy_score(y_pred, y_valid))
        #print(classification_report(y_valid, y_pred,target_names=['-1.0','0.0','1.0']))

    def TrainRandomForest(self):
        #X_train, X_valid, y_train, y_valid = train_test_split(self.trainingVector, self.classVector, test_size = 0.1, shuffle = True)
        clf = RandomForestClassifier(n_estimators=10)
        scores = cross_validate(clf,self.trainingVector, self.classVector, scoring=make_scorer(self.classification_report_with_accuracy_score),
                                cv=10, return_train_score=False)

        print("Scores for Random Forest.....")
        print(scores)

        #clf.fit(X_train,y_train)



        #y_pred = clf.predict(X_valid)
        #print('Random forest: accuracy %s' % accuracy_score(y_pred, y_valid))
        #print(classification_report(y_valid, y_pred,target_names=['-1.0','0.0','1.0']))
	
    def TrainNeuralNet(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.trainingVector, self.classVector, test_size = 0.1, shuffle = True)
        numberOfClasses = 3
        batchSize = 64
        nbEpochs = 10

        Y_train = np_utils.to_categorical(y_train, numberOfClasses)
        Y_valid = np_utils.to_categorical(y_valid, numberOfClasses)


        #print(Y_train)
        #print(y_train)
        #quit()
        model = Sequential()
        model.add(Dense(1000,input_shape=(10000,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(numberOfClasses))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'] )
        print(model.summary())
        model.fit(X_train, Y_train,validation_data=(X_valid,Y_valid),batch_size=batchSize,epochs = nbEpochs, verbose=1, )

        #y_pred = model.predict(X_valid, batch_size=batchSize)
        #print(y_pred)
        #print(y_valid)
        #print('Neural Net: accuracy %s' % accuracy_score(y_pred, y_valid))
        #print(classification_report(y_valid, y_pred,target_names=['-1.0','0.0','1.0']))

