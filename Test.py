from sklearn.externals import joblib
import xlrd

from Tweet import Tweet
from Database import Database
from collections import Counter
import numpy as np
import pickle

current_row = 1

    # path to the file you want to extract data from
src = r'D:\CS 583\Project 2\Test.xlsx'

book = xlrd.open_workbook(src)

    # select the sheet that the data resids in
work_sheet = book.sheet_by_index(1)

    # get the total number of rows
num_rows = work_sheet.nrows - 1
testDatabase=Database();
#obamaDatabase = Database()


while current_row <= num_rows:
    tweet_text = work_sheet.cell_value(current_row,1 )
    #print(tweet_text)
    tweet_class= None
    # print ("Tweet going in",tweet_text)
    tweet= Tweet(tweet_text,tweet_class);
    tweet.TweetID = work_sheet.cell_value(current_row,0 )
    tweet.Cleanself();
    tweet.TagPOS();
            #print("After POS",tweet.WordList);
    tweet.lemmatize();
            #print("After Lemmatized",tweet.stemmedList);
            #tweet.Stemself();


    testDatabase.add(tweet);
    current_row+=1;

stemmedTexts=[o.stemmedList for o in testDatabase.Table]

loaded_vectorizer = pickle.load(open("RomneyVectorizer.pickle", "rb"))
        #vectorizer.fit(stemmedTexts)
testVector = loaded_vectorizer.transform(stemmedTexts).todense()
loaded_model = pickle.load(open("RomneyLinearSVM.pickle", "rb"))
result = loaded_model.predict(testVector)

print(result)

ouputFile = "David_Torres_Mobashir_Sadat_Romney.txt"
with open(ouputFile, 'a') as out:
    i=0
    out.write("Tweet_ID;;Predicted Class Label\n")
    for tweet in testDatabase.Table:
        out.write(str(int(tweet.TweetID))+';;'+str(int(result[i]))+'\n')
        i = i+1
