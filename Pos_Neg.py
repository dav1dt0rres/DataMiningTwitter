from nltk.stem import WordNetLemmatizer




class Positive():
    """Extract features from each document for DictVectorizer"""
    def __init__(self,lines_pos):
        self.posList=[]

        self.wnl=wnl = WordNetLemmatizer()

        for line in lines_pos:
            word=line.strip('\t').strip("").strip('\n').rstrip().lower()
            self.posList.append(word)


    def fit(self,stemmedTexts,y=0):

        return self

    def transform(self, stemmedLists):
        print("INside Transform for PosNeg!!")
        temp=[]
        for wordlist in stemmedLists:
            pos_count=0;

            for word in wordlist:
                #print("word going in",word)
                if self.foundPos(word):
                    #input("Foudn positive")
                    pos_count=+1;

            temp.append(pos_count)

        final=[[i] for i in temp]

        return final
    def foundPos(self,word):
        for pos in self.posList:

            if self.wnl.lemmatize(word)==pos:
                return True;
        return False;








class Negative():
    def __init__(self,lines_neg):

        self.negList=[]
        self.wnl=wnl = WordNetLemmatizer()

        for line in lines_neg:
            word=line.strip('\t').strip("").strip('\n').rstrip().lower()
            self.negList.append(word)

    def fit(self,stemmedTexts,y=0):

        return self

    def transform(self, stemmedLists):
        print("INside Transform for PosNeg!!")
        temp=[]
        for wordlist in stemmedLists:

            neg_count=0;
            for word in wordlist:


                if self.foundNeg(word):
                    #input("Foudn negative")
                    neg_count=+1;
            temp.append(neg_count)

        final=[[i] for i in temp]

        return final

    def foundNeg(self,word):

        for neg in self.negList:

            if self.wnl.lemmatize(word)==neg:
                return True;
        return False;
