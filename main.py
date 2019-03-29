import xlrd


from Tweet import Tweet
from Database import Database

if __name__ == '__main__':



    current_row = 2
    sheet_num = 1
    input_total = 0
    output_total = 0

    # path to the file you want to extract data from
    src = r'C:\Users\david\Downloads\trainingObamaRomneytweets.xlsx'

    book = xlrd.open_workbook(src)

    # select the sheet that the data resids in
    work_sheet = book.sheet_by_index(1)

    # get the total number of rows
    num_rows = work_sheet.nrows - 1
    database=Database();

    while current_row < num_rows:
        tweet_text = work_sheet.cell_value(current_row,3 )
        tweet_class= work_sheet.cell_value(current_row,4 )
       # print ("Tweet going in",tweet_text)
        tweet= Tweet(tweet_text,tweet_class);
        tweet.Cleanself();
        #Mobis's Lemmatizer
        #Mobis's Stop/Too Commonn Filter

        tweet.Stemself();

        #tweet.SpellCheck();
        database.add(tweet);
        #print("Size so far",database.getSize());
        current_row+=1;

    #Vectorizing
    print("Writing on Text")
    database.write();#1->print just Full text, #2--> pring Stemmed data,-->#3 basically (1) AND sPELL CHECK
    exit(0);