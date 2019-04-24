



class KeySignals():
    """Extract features from each document for DictVectorizer"""
    def __init__(self):

        self.SwearList=['shit','fuck','nigger','bitch','monkey','fucking','shitty','cracker']
    def fit(self,stemmedTexts,y=0):
        print("INside fit!!!")
        return self

    def transform(self, stemmedLists):
        print("INside Transform!!",stemmedLists)
        temp=[]
        for wordlist in stemmedLists:
            trigger=False
            for word in wordlist:
                if self.foundSwear(word):
                    trigger=True
                    temp.append(float(1))
                    break;
            if (trigger==False):
                temp.append(float(0))

        final=[[i] for i in temp]
        #print("temp",final)

        return final
    def foundSwear(self,word):
        for swear in self.SwearList:
            if word==swear:
                return True
        return False

