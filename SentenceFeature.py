# Sentences as features

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import numpy as np

def readString(fileName):
    """
    Function that takes in a file path and returns a list of all the lines in the file as a string
    Parameters:
        fileName - A string containing a path to a file  
    """
    splitData = []
    with open(fileName, 'r') as myfile:
        data=myfile.read()
        splitData=data.splitlines()
    assert(splitData != None)
    assert(splitData != [])
    return splitData

# TFIDF feature extractor and remove stopwords
    
def feature_extractor(data):
    """
    Function that vectorizes and converts the incoming data into a matrix
    Parameters:
        data - contains sentences from good and bad deals
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_features = 338, max_df=0.5,stop_words='english')
    mat = vectorizer.fit_transform(data)
    return mat  

# Linear SVM Classifier

def classify(X_train,Y_train, X_test):
    clf = LinearSVC()
    clf.fit(X_train.toarray(), Y_train)
    Y_pred = clf.predict(X_test.toarray())
    return Y_pred

# KNN Classifier    
    
def KNN(X_train,Y_train, X_test):
    clf = KNeighborsClassifier()
    clf.fit(X_train.toarray(), Y_train)
    Y_pred = clf.predict(X_test.toarray())
    return Y_pred
    
# Function for cross validation
    
def crossValidation(X_train,Y_train):
    """
    Function to perform cross validation on the dataset
    """
    X_train, X_test, Y_train, y_test = cross_validation.train_test_split(X_train.toarray(), Y_train, test_size=0.4)
    
    # Using SVM
    clf = LinearSVC()
    clf.fit(X_train, Y_train)
    Y_predSVC = clf.predict(X_test)
    accuracySVC=clf.score(X_test,y_test)
    
    # Using KNN
    clf1 = KNeighborsClassifier()
    clf1.fit(X_train, Y_train)
    Y_predKNN = KNN(X_train, Y_train, X_test)
    clf1.predict(X_test)
    accuracyKNN=clf1.score(X_test,y_test)
    return accuracySVC,accuracyKNN 

# Function to check predicted SVM accuracy with the manual prediction

def checkPrediction(listName):
    """
    Checking the predicted values against the actual values
    """
    countGood = 0
    countBad = 0   
    for sen in test:
        for prediction in listName:
            if sen in goodList and prediction == 1:
                countGood += 1
            if sen in badList and prediction == 0:
                countBad += 1
        totalCorrect = countGood + countBad              
    totalLength = len(goodList) + len(badList)
    classifierAccuracy = (float(totalCorrect)/totalLength)*100
    return classifierAccuracy

# Function to compare correct prediction of SVM and KNN

def compareSVMandKNN(predictionSVM,predictionKNN):
    """
    Comparison of SVM and KNN results
    """
    count=0
    mismatch=0
    for item in range(len(predictionSVM)):
        if predictionSVM[item]==predictionKNN[item]:
            count=count+1
        else:
            mismatch=mismatch+1
    agreeingResultPercentage=(float(count)/(mismatch+count))*100
    return agreeingResultPercentage


# Read text file to string
    
good=list(readString("good_deals.txt"))
bad=list(readString("bad_deals.txt"))
test=list(readString("test_deals.txt"))

# Call feature extractor function and built training set and test set

X_train=feature_extractor(good+bad)
bad_labels = [0]*len(bad)
good_labels = [1]*len(good)
X_test=feature_extractor(test)
Y_train = np.asarray(good_labels+bad_labels)

# Call SVM and KNN classifier to predict accuracy

accuracy_SVM, accuracy_KNN=crossValidation(X_train,Y_train)
print "Cross Validation Accuracy with KNN(60/40):  ",accuracy_KNN
print "Cross Validation Accuracy with Linear SVM(60/40):  ",accuracy_SVM                         
predictionSVM=classify( X_train,Y_train, X_test) 
predictionSVMList =  list(predictionSVM)   
predictionKNN=KNN( X_train,Y_train, X_test)
predictionKNNList = list(predictionKNN)
  
# Check predicted SVM accuracy with the manual prediction

SVMpredict=checkPrediction(predictionSVMList)
print "Accuracy of SVM with Test Data is:", SVMpredict
KNNPredict=checkPrediction(predictionKNNList)
print "Accuracy of KNN with Test Data is:", KNNPredict

# Compare correct prediction of SVM and KNN

Accuracy=compareSVMandKNN(predictionSVM,predictionKNN)
print "Correct predicted accuracy of both SVM and KNN:" , Accuracy

    
