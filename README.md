# Assignment-5-FakeNews-Detection:

1. For assigment 5 we are creating a ML pipeline for fake nnews detection.

## Task1:

1. For task one we load in the csv file and create a temp view before runing some basic queries, outputting to a new csv file with the dataframe.

## Task2:

For task two we processed the text by converting to lowercase, splitting the text into words with a tokenizer, and removing stopwords before outputting to a new csv.

## Task3:

For task three we use hasingtf and idf on te tokenized text, changing the true and fake labels to 0s and 1s before assembling and saving to csv.

## Task4:

For task four we were supposed to train the model by splitting into 80 and 20 test sets before using logistic regression and generating predictions, saving once again to a new csv. I had the most issues on this part and I'll describe below.

## Task5:

For task five we were supposed to use multiclassclassificationevaluator to get the accuracy and f1 score before returning a new csv with both, but I never got to try running tis part since part 4 was giving me so much issue.

# Issues

I started trying to run my created task 4 on my own, intitally just getting issues with the dataset format. I then went back and tried anoher rendition of task 3 to clean the output up a bit, but even after trying on my own and using chatgpt/copilot to try sifting through the errors and tweaking the code it still didnt work after 20+ attempts and a couple hours of tries. I decided just to leave it as is and at least try jotting something down for task 5 even though there's no way to really test it withouit task 4s output csv. I'll try speaking with a TA to see if I can get it fixed, but for now I'm just going to turn in this attempt since I have work pretty much every day until exams start. Thanks