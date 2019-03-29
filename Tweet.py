from __future__ import print_function

import re
from Spell import Spell
import string
import nltk

from nltk.stem.porter import *
from nltk.stem import *
from nltk.corpus import wordnet
class Tweet:

    #Queue contains a list of lines only for that firm. (Function Excel)

    def __init__(self,tweet_text,tweet_class):

        self.Dictionary={
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "I'd": "I would",
            "I'd've": "I would have",
            "I'll": "I will",
            "I'll've": "I will have",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it'd": "it had",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it has",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she had",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that had",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there had",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they had",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "wwho will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you shall have",
            "you're": "you are",
            "you've": "you have"
        }
        self.stemmer = PorterStemmer()
        self.WordList = []
        self.TweetText=tweet_text
        self.Class=tweet_class;
        self.stemmedList=[]
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    def SpellCheck(self):
        spell = Spell()

        # find those words that may be misspelled

        index=0;
        for word in self.WordList:
            # Get the one `most likely` answer
            correction=spell.correction(word.lower())
            print("word",word)
            print("correction",correction)


            if word.lower()!=correction:
                if (word.lower()!='mitt' and word.lower()!='obama'and word.lower()!='romney'):

                    answer=input("Correction occured; 1 to replace, 2 to keep")
                    if (answer=='1'):
                        self.WordList[index]=correction
                        print(self.WordList)
                        input();

            index +=1;


    def Stemself(self):
        self.stemmedList = [self.stemmer.stem(plural) for plural in self.WordList]
        #print("Stemmed List",self.stemmedList);

    def getWordList(self):
        printable = set(string.printable);
        self.WordList=filter(lambda x: x in printable, self.WordList)
        print("WordList returing",self.WordList)
        return self.WordList
    def removePunctuations(self):
        self.TweetText = self.TweetText.replace("'s","")
        #self.TweetText = self.TweetText.replace("'","")
        self.TweetText = self.TweetText.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
        
    def Cleanself(self): #simply compare the overall number of lines in both Lists
        index=0;
        self.removePunctuations()
        self.WordList=nltk.word_tokenize(self.TweetText);
        #print("wordlist",self.WordList)
        for word in self.WordList:
            #print("before",word)
            word=word.replace('</a>', '').replace('</e>', '').replace('<e>','').replace('<a>', '')

            regex= re.compile('[$;%&*"“”#,@.!():?]')
            word=regex.sub('', word)
            word=word.lower();
            #print("after",word)


            self.WordList[index]=word

            if self.WordList[index].isdigit():
                self.WordList.pop(index)

            elif '@' in self.WordList[index]:
                self.WordList.pop(index);
                #print(self.WordList)
                #input("removing @")

            elif 'http' in self.WordList[index]:
                self.WordList.pop(index);


            elif (None!=self.Dictionary.get(self.WordList[index].lower())):
                temp_word=self.WordList.pop(index)
                for word_1 in self.Dictionary.get(temp_word.lower()).split():
                    self.WordList.insert(index,word_1)



            index+=1

        #print(self.WordList)


        #for word in self.WordList:
            #print("List in a",word);
    def TagPOS(self):
        try:
            self.WordList = nltk.pos_tag(self.WordList)
        except:
            print('errorTagPos')
        
        
    def lemmatize(self):
        tokens = []
        for token in self.WordList:
            try:
                if token[1].startswith('N'):
                    tokens.append(self.lemmatizer.lemmatize(token[0],pos=wordnet.NOUN))
                elif token[1].startswith('J'):
                    tokens.append(self.lemmatizer.lemmatize(token[0],pos=wordnet.ADJ))
                elif token[1].startswith('R'):
                    tokens.append(self.lemmatizer.lemmatize(token[0],pos=wordnet.ADV))
                elif token[1].startswith('V'):
                    tokens.append(self.lemmatizer.lemmatize(token[0],pos=wordnet.VERB))
                else:
                    tokens.append(self.lemmatizer.lemmatize(token[0],pos=wordnet.NOUN))
            except:
                print("Erros lemmatizer")

        self.stemmedList = tokens