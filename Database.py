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
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import nltk
from sklearn.model_selection import StratifiedKFold


import xlrd
from Tweet import Tweet
from collections import Counter

import pickle

class Database:

    #Queue contains a list of lines only for that firm. (Function Excel)

    def __init__(self):
        #print("Initialization of Join")
        self.trainingVector=[];
        self.Table=[];
        self.classVector=[]
        self.vectorizer=None;
        self.classification_report = []
        self.accuracy = []

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

    def tokenizer(self,x):
        return x
    def preprocessor(self,x):
        return x

    def Vectorizeself(self,vectorizerFile):
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

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word',tokenizer=self.tokenizer,preprocessor=self.preprocessor,token_pattern=None,stop_words=wordsToIgnore)
        
        stemmedTexts=[o.stemmedList for o in self.Table]
        vectorizer.fit(stemmedTexts)
        self.vectorizer = vectorizer
        self.trainingVector = vectorizer.transform(stemmedTexts).todense()
        print(self.trainingVector.shape)
       # quit()
        #print("after transform",self.trainingVector.toarray())
        
        print(self.trainingVector.shape)
        pickle.dump(vectorizer, open(vectorizerFile, "wb"))
       # return vectorizer
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

        print (classification_report(y_true, y_pred))
         # print classification report
        self.classification_report.append(classification_report(y_true,y_pred, output_dict=True))
        self.accuracy.append(accuracy_score(y_true, y_pred))
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

    def TrainLinearSVM(self,modelFile):
        #X_train, X_valid, y_train, y_valid = train_test_split(self.trainingVector, self.classVector, test_size = 0.1, shuffle = True)
        scoring = ['precision_macro', 'recall_macro','f1_weighted']
        clf = OneVsOneClassifier(LinearSVC(random_state=0))
        scores = cross_validate(clf,self.trainingVector, self.classVector, scoring=make_scorer(self.classification_report_with_accuracy_score),
                                cv=10, return_train_score=False)

        model_to_save = OneVsOneClassifier(LinearSVC(random_state=0)).fit(self.trainingVector, self.classVector)
        filename = modelFile
        pickle.dump(model_to_save, open(filename, 'wb'))

        print("Scores for LinearSVM.....")
        print(scores)
        return scores
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
    def TrainLogisticRegression(self):
        clf = RandomForestClassifier(n_estimators=10)
        scores = cross_validate(clf,self.trainingVector, self.classVector, scoring=make_scorer(self.classification_report_with_accuracy_score),
                                cv=10, return_train_score=False)

        print("Scores for Random Forest.....")
        print(scores)


    def EvaluateNeuralNet(self):
        X = self.trainingVector
        Y = map(int, self.classVector)

        skf = StratifiedKFold(n_splits=10,shuffle=True)
        skf.get_n_splits(X, Y)
        print(skf)
        for train_index, test_index in skf.split(X, Y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
        #for i, (train, test) in enumerate(skf):



    def TrainNeuralNet(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.trainingVector, self.classVector, test_size = 0.1, shuffle = True)
        numberOfClasses = 3
        batchSize = 64
        nbEpochs = 2

        Y_train = np_utils.to_categorical(y_train, numberOfClasses)
        Y_valid = np_utils.to_categorical(y_valid, numberOfClasses)

        model = Sequential()
        model.add(Dense(1000,input_shape=(X_train.shape[1],)))
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

        model.fit(X_train, Y_train,validation_data=(X_valid,Y_valid),batch_size=batchSize,epochs = nbEpochs, verbose=1, )

        y_pred = model.predict(X_valid, batch_size=batchSize)
        y = y_pred.argmax(axis=-1)

            #print(y)
            #print(y_valid)

        predictedY = []
        for val in y:
            if val == 0:
                predictedY.append(0)
            elif val == 1:
                predictedY.append(1)
            elif val == 2:
                predictedY.append(-1)


        print('Neural Net: accuracy %s' % accuracy_score(predictedY, y_valid))
        print(classification_report(y_valid, predictedY,target_names=['-1.0','0.0','1.0']))    

    def getAverageScores(self):
        precisionPositive = 0
        precisionNegative = 0
        precisionNeutral = 0
        recallPositive = 0
        recallNegative = 0
        recallNeutral = 0
        fscorePositive = 0
        fscoreNegative = 0
        fscoreNeutral = 0
        supportPositive = 0
        supportNegative = 0
        supportNeutral = 0

        for report in self.classification_report:
            report_dict = dict(report)
            precisionPositive = precisionPositive + report_dict['1.0']['precision']
            precisionNegative = precisionNegative + report_dict['-1.0']['precision']
            precisionNeutral = precisionNeutral + report_dict['0.0']['precision']

            recallPositive = recallPositive + report_dict['1.0']['recall']
            recallNegative = recallNegative + report_dict['-1.0']['recall']
            recallNeutral = recallNeutral + report_dict['0.0']['recall']

            fscorePositive = fscorePositive + report_dict['1.0']['f1-score']
            fscoreNegative = fscoreNegative + report_dict['-1.0']['f1-score']
            fscoreNeutral = fscoreNeutral + report_dict['0.0']['f1-score']

            supportPositive = supportPositive + report_dict['1.0']['support']
            supportNegative = supportNegative + report_dict['-1.0']['support']
            supportNeutral = supportNeutral + report_dict['0.0']['support']




        print('Average Precision 1:',precisionPositive/10,'-1:',precisionNegative/10,'0',precisionNeutral/10)
        print('Average Recall 1:',recallPositive/10,'-1:',recallNegative/10,'0',recallNeutral/10)
        print('Average f1-score 1:',fscorePositive/10,'-1:',fscoreNegative/10,'0',fscoreNeutral/10)
        print('Average support 1:', supportPositive/10,'-1:',supportNegative/10,'0',supportNeutral/10)

        accuracyTotal = 0
        for accuracy in self.accuracy:
            accuracyTotal = accuracyTotal + accuracy
        print('Average accuracy: ',accuracyTotal/10)



        
       
            
       


            

