from sklearn.externals import joblib
import xlrd
import csv

from Tweet import Tweet
from Database import Database
from collections import Counter
import numpy as np
import pickle

current_row = 1

    # path to the file you want to extract data from
src = r'D:\CS 583\DataMiningTwitter\Obama_Romney_Test_dataset_NO_label\Obama_Test_dataset_NO_Label.csv'

#book = xlrd.open_workbook(src)

    # select the sheet that the data resids in
#work_sheet = book.sheet_by_index(0)

    # get the total number of rows
#num_rows = work_sheet.nrows - 1
testDatabase=Database();
#obamaDatabase = Database()


with open(src, encoding="ISO-8859-1") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    i=0
    for row in readCSV:
        if i>=1:
            tweet_text = row[1]
    #print(tweet_text)
            tweet_class= None
            # print ("Tweet going in",tweet_text)
            tweet= Tweet(tweet_text,tweet_class);
            tweet.TweetID = row[0]
            tweet.Cleanself();
            tweet.TagPOS();
                    #print("After POS",tweet.WordList);
            tweet.lemmatize();
                    #print("After Lemmatized",tweet.stemmedList);
                    #tweet.Stemself();
            print(tweet)

            testDatabase.add(tweet);
        i = i+1

#print(len(testDatabase.Table))
#while current_row <= num_rows:
 #   tweet_text = work_sheet.cell_value(current_row,1 )
    #print(tweet_text)
#    tweet_class= None
    # print ("Tweet going in",tweet_text)
#    tweet= Tweet(tweet_text,tweet_class);
#    tweet.TweetID = work_sheet.cell_value(current_row,0 )
#    tweet.Cleanself();
#    tweet.TagPOS();
            #print("After POS",tweet.WordList);
#    tweet.lemmatize();
            #print("After Lemmatized",tweet.stemmedList);
            #tweet.Stemself();


 #   testDatabase.add(tweet);
  #  current_row+=1;

stemmedTexts=[o.stemmedList for o in testDatabase.Table]

loaded_vectorizer = pickle.load(open("ObamaVectorizer.pickle", "rb"))
        #vectorizer.fit(stemmedTexts)
testVector = loaded_vectorizer.transform(stemmedTexts).todense()
loaded_model = pickle.load(open("ObamaLinearSVM.pickle", "rb"))
result = loaded_model.predict(testVector)

print(result)

ouputFile = "David_Torres_Mobashir_Sadat_Obama.txt"
with open(ouputFile, 'a') as out:
    i=0
    out.write("Tweet_ID;;Predicted Class Label\n")
    for tweet in testDatabase.Table:
        out.write(str(int(tweet.TweetID))+';;'+str(int(result[i]))+'\n')
        i = i+1
