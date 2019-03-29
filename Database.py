import re
import sklearn
import string
from collections import Counter

import nltk


class Database:

    #Queue contains a list of lines only for that firm. (Function Excel)

    def __init__(self):
        print("Initialization of Join")

        self.Table=[];

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

                    print("Printing Word",self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word)))
                    OpenFile3.write(self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word))+";"+"\t")
                OpenFile3.write(str(tweet.Class)+";"+"\n")
        elif(bit==2):#Stemmed
            for tweet in self.Table:
                for word in tweet.stemmedList:
                    print("Printing Word",self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word)))
                    OpenFile3.write(self.LastClean(re.sub(r'[^\x00-\x7f]',r'', word))+";"+"\t")

                OpenFile3.write(str(tweet.Class)+";"+"\n")

        elif(bit==3):#Spell Checker and Full(Cleaned)
            print("Printing Spellchecker")

        OpenFile3.close()

    def add(self,Tweet):

        self.Table.append(Tweet)
    def getSize(self):
        return len(self.Table)

    def Vectorizeself(self):
        print("Inside Vectorizing")
        #Vectorizing
        rare_words = self.get_rare_words(1)
        stopwords=nltk.corpus.stopwords.words('english')
        wordsToIgnore = list(set(stopwords + rare_words))

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word',tokenizer=lambda x: x,preprocessor=lambda x: x,token_pattern=None,stop_words=wordsToIgnore)

        stemmedTexts=[o.stemmedList for o in self.Table]
        print("after FIt",vectorizer.fit(stemmedTexts))

        trainingVector = vectorizer.transform(stemmedTexts)
        print("after transform",trainingVector.toarray())

        print(trainingVector.shape)
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