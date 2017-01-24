# Words as features

import nltk
import enchant
import random
import sys
from nltk.corpus import stopwords

# Read text file into String

def readString(fileName):
    """
    Function to take in a file path and read all the contents of the file into a string
    """
    if (fileName == '' || fileName is None):
        sys.exit("Cannot open file as the file path is either not a string or is empty string. Exiting")
    with open(fileName, 'r') as myfile:
        data=myfile.read()
    assert(data != '')
    assert(data != None)
    return data

# Tokenize and remove digits from the string
    
def removeDigits(goodDealRead):
    """
    Helper function to tokenize and remove all numeric strings since they cannot be assessed 
    """
    tokens=nltk.wordpunct_tokenize(goodDealRead)          
    text=nltk.Text(tokens)
    words=[w.lower()for w in text if w.isalpha()]
    assert(words != [])
    return words

# Check English Dictionary to keep only valid English words
    
def checkDictionary(words):
    """
    Using python enchant library to get a list of words from English language
    """
    EnglishList=[]
    d = enchant.Dict("en_US")                                           
    for i in words:
        if d.check(i) is True:
            EnglishList.append(i)
    assert(EnglishList != [])
    return EnglishList

def removeStopwords(EnglishList):
    """
    Function to remove stop words from the list of words 
    """
    stopWords=set(stopwords.words('english'))
    filtered= filter(lambda word:not word in stopWords, EnglishList) 
    assert(filtered != [])
    return filtered

# Function to read the text file into a list

def readFile(text):
    """
    Function to read file into text, remove digits, filter for english words and remove stop words
    to finally return a filtered list of useful english words
    """  
    fileText=readString(text)
    words=removeDigits(fileText)
    EnglishList=checkDictionary(words)
    filtered=removeStopwords(EnglishList)
    assert(filtered != []ss)
    return filtered

# Cross Validation accuracy calculation

def crossValidation(mergedList):
    """
    Function to perform cross validation of the classifier on the dataset and return the accuracy of the Naive Bayes Classifier
    Parameters:
        mergedList - A list of tuples of the form [({"Word":"discount"}, 1),({"Word":"someBadWord"},0)]
    """
    random.shuffle(mergedList)
    train_set=mergedList[:205]
    test_set = mergedList[206:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    accuracy_NB=nltk.classify.accuracy(classifier, test_set)
    return accuracy_NB

# Naive Bayes Classifier returning accuracy of classification

def testClassifier(mergedList,testList):
    """
    Function that takes in a list of words present in both good and bad deals 
    and a list of test deals for classification
    Parameters:
        mergedList - A list of tuples of the form [({"Word":"discount"}, 1),({"Word":"someBadWord"},0)]
        from the labeled data
        testList - A list of tuples of the form [({"Word":"discount"}, 1),({"Word":"someBadWord"},0)]
        from unlabeled data
    """
    random.shuffle(mergedList)
    train_set, test_set = mergedList, mergedTestList
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print classifier.show_most_informative_features(20)
    return nltk.classify.accuracy(classifier, test_set)

# Reading and creating list of tuples for good deals
good_deals_words=readFile("good_deals.txt")
listOfTuplesGood = [({"Word":item}, 1) for item in good_deals_words]

# Reading and creting list of tuples for bad deals
bad_deals_words=readFile("bad_deals.txt")
listOfTuplesBad = [({"Word":item}, 0) for item in bad_deals_words]

# Combining list of terms for good and bad deals
mergedList=listOfTuplesGood+listOfTuplesBad

# Reading in test deal words for populating the corpus
test_deal_words=readFile("test_deals.txt")
testList = [({"Word":item}) for item in test_deal_words]

# Reading in bad deals from test set
test_deal_words_bad=readFile("test_bad.txt")
testBad=[({"Word":item},0) for item in test_deal_words_bad]

# Reading in good deals from test set
test_deal_words_good=readFile("test_good.txt")
testGood = [({"Word":item},1) for item in test_deal_words_good]

# Combining and creating a list of words combined from the good and bad test set
mergedTestList=testGood+testBad

# Checking the results with cross validation
accuracy_NB=crossValidation(mergedList)
print "Cross Validation Accuracy with Naive Bayes(60/40):  ", accuracy_NB

# Testing the classifier with the test set
accuracyTest=testClassifier(mergedList,mergedTestList)
print" Test Accuracy with Naive Bayes:  ", accuracyTest










