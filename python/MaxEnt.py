import sys
import getopt
import os
from fractions import Fraction

import numpy as np
import math
import operator
import nltk
import re
import nltk.classify.util
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
np.set_printoptions(threshold='nan')

global BOW
global maxWords
global negBOW
global posBOW
global features
global X
global weights
global positiveWords
global negativeWords
global XMap

XMap={'A':'B'}
BOW = set()
#these are maps of word with corresponding counts
BOW1 = {}
negBOW = {}
posBOW = {}
maxWords=20000
positiveWords={}
negativeWords={}

stemmer = PorterStemmer()





class Maxent:
  #bag of word matrix. Its a SET, and hence contains only unique elements

  stopwords = nltk.corpus.stopwords.words('english')

  #pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')


  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Maxent initialization"""
    
    self.numFolds = 10

  class WordCount:
    def __init__(self):
      self.word=''
      self.count=0


  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Maxent classifier
  """
  1. Each word in all the documents are a feature
  2. Associate a weight for all the features
  3. Try to change weight using Gradient Descent algorithm to associate a weight to all the features
  4. Compute the actual probability of a document to occur in a class c using the weights and probability function
    
  """

  #TODO put all MAXENT related classifiers, etc. Here we have to actually classify based on maxent classifier

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    X = findFeatureMatrixForDataSet(words,None)
    probPos=summation(1,X)
    probNeg=summation(0,X)
    if probPos+probNeg>=0:
      return 'pos'
    return 'neg'

    # X= findFeatureMatrixForDataSet(words)
    # probPos=summation(1,X)
    # print "prob Pos ", probPos
    # probNeg=summation(0,X)
    # print "prob Neg ", probNeg
    # #print "probPos ",probPos
    # #print "probNeg ",probNeg
    # if(probPos>probNeg):
    #   return 'pos'
    # return 'neg'

    # Write code here


#for training data
  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Maxent class.
     * Returns nothing
    """

    # Write code here

    for r1 in words:
       if r1 not in self.stopwords:
         if re.match('\w',r1) and len(r1)>1:
            r=stemmer.stem(r1)
            if klass=='pos':
              if r not in posBOW:
                posBOW[r.lower()]=1
              else:
                posBOW[r.lower()]=posBOW[r.lower()]+1
            elif klass =='neg':
              if r not in negBOW:
                negBOW[r.lower()]=1
              else:
                negBOW[r.lower()]=negBOW[r.lower()]+1

    pass

  """This function is for training the classifier. This is called after negative and positive documents are added to the split"""
  #TODO train the classifier



  def train(self, splits, epsilon, eta, lambdaa):
      """
      * TODO
      * iterates through data examples
      """
      global BOW
      global weights
      global features
      global BOW1
      global posBOW
      global negBOW

      for split in splits:
        for example in split.train:
          if len(posBOW)<maxWords or len(negBOW)<maxWords:
            words = example.words
            self.addExample(example.klass, words)
          else:
            break

      i=0

      posBOW = sorted(posBOW,key=posBOW.get, reverse=True) #sort the map of BOW1 based on value of each key in descending order
      negBOW = sorted(negBOW, key=negBOW.get,reverse=True)  # sort the map of BOW1 based on value of each key in descending order
      if maxWords>0 and maxWords<len(posBOW):
        list1 = posBOW[:maxWords / 2]
      else:
          list1=list(posBOW)
      if maxWords>0 and maxWords<len(negBOW):
        list1=list1+list(negBOW[:maxWords / 2])
      else:
          list1=list1+list(negBOW)

      BOW=set(list1)
      print "BOW "
      print BOW
      global posFeatures
      global negFeatures
      #TODO now we have to train our classifiers
      n=len(BOW)
      #print "Length is ",n
      features = []
      negFeatures=[]
      posFeatures=[]

      features=list(BOW)
      features.append('1')
      negFeatures=list(negBOW)
      posFeatures=list(posBOW)

      global length
      length=n+1
      X = np.zeros(((n+1),1))   #fills all X with 0s initially. Then start changing X based on input we are talking about in the document
      weights = np.zeros(((n+1),1))


      weights=np.random.rand(n+1,1)/10

      for i in range(n/2,n+1):
           weights[i]=-weights[i]

       # for i in range(0,n):
       #   r=features[i]
       #   weights[i,0] = -np.random.random()

      gradientDescent(splits,epsilon,eta,lambdaa)
      #now we have weights, so we have to find out probability of a dataset belonging to a class




  # END TODO (Modify code beyond here with caution)
  #############################################################################


  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  """Train split, reads all the files in pos and neg directories, and then does the following:
  1. split - stores all the words array in split
      split.train.append(example)
      where example = 'example.words' and 'example.klass' """
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
     # print example.words.__str__() +" \n ******************************************* "+example.klass
      split.train.append(example)   #words will be appneded along with the class pos or neg to the split
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)  #words will be appneded along with the class pos or neg to the split
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a list of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  




def test10Fold(args):
  pt = Maxent()
  epsilon=0.01
  eta=0.01
  lambdaa=2
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  classifier = Maxent()
  classifier.train(splits, epsilon, eta, lambdaa)

  #print "BOW length ", len(BOW)
  #print weights.transpose()
  #print features
  print "Weights of Features"
  for i in range(0,len(weights)-1):
    print features[i]," =  ",weights[i]

  for split in splits:
    accuracy = 0.0
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0
    if len(split.test)!=0:
      accuracy = accuracy / len(split.test)
      avgAccuracy += accuracy
      print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
      fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyDir(trainDir, testDir,iter):
  epsilon = 0.001
  eta = 0.001
  lambdaa = 0.001
  classifier = Maxent()
  trainSplit = classifier.trainSplit(trainDir)   #simply partitions data into positive and negative files for training directory
  #trainSplit contains set of array as {{word: [w1,w2,w3,...],klass:pos}}

  iterations = int(iter)  # changing string iter to integer
  classifier.train(trainSplit, epsilon, eta, lambdaa)  #TODO no training has been defined yet
  # after classifier.train() step, using trainSplit, our classifier will be trained and then it will be used to test, testDir in next step

  testSplit = classifier.trainSplit(testDir)  #again repeating same thing for testing directory
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0

  #Training data has created the model first, now we are predicting on test data for understanding the accuracy.
  for example in testSplit.train:  #for each example in test directory , we check if the guess is same as the actual directory
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy

#**********************Gradient Descent and others*********************

#returns the matrix for gradient descent computation
# matrix like ([1/(1+e-thatax)]-y)*X
def summation(y,X):
  #just find summation over lambda, and then take power of exponent
    global weights
  # power=np.dot(weights.transpose(),X)  # theta transpose * X where theta is weight
  # power1=power[0,0]  #first element
  # expPower=float(pow(2.718,-power1))
  # value=float(((1)/(1+expPower)))
  # value=value-y
  #lambda1 -> weight in pos class
  #f1 -> frequency of word in document
  #X -> the actual document represented in terms of frequency for each feature. We only select features which are of positive class
    product=1
    global posFeatures
    posFeaturesMatrix=np.zeros(((length),1))
    for i in range(0,length/2):
      posFeaturesMatrix[i]=1
    negFeaturesMatrix = np.zeros(((length), 1))
    for i in range(length/2,length):
      negFeaturesMatrix[i]=1



    if y==1:
      posFeaturesMatrix=np.multiply(posFeaturesMatrix,X)
      arr=np.dot(weights.transpose(),posFeaturesMatrix)
    else:
      negFeaturesMatrix=np.multiply(negFeaturesMatrix,X)
      arr=np.dot(weights.transpose(),negFeaturesMatrix)
    return float(arr[0,0])

def gradientDescent(splits,epsilon,eta,lambdaa):
  global weights
  delta=1000000
  maxCount=500
  count=1

  print "Weights ", weights
  print "features ",features
  #while delta>epsilon:
  print "*********gradient descent started***********"
  while delta>epsilon and count <maxCount :
    #print "Iteration ",delta
    i=0
    for split in splits:
      m=len(split.train)
      sumMatFinal=np.zeros(((length),1))
      currentWeights=np.zeros(((length),1))
      for example in split.train:
        if count>maxCount:
          break
        words = example.words  #each document
        X = findFeatureMatrixForDataSet(words,i)
        i = i + 1
        posSum=summation(1,X)
        negSum=summation(0,X)
        dif = negSum + posSum
        #dif = negSum - posSum
        #posSum = pow(2.718, -negSum-posSum)
        #negSum=pow(2.718,posSum+negSum)

        prob=1.00000
        if example.klass == 'pos':
          y=1
          prob=float(1/(1+pow(2.718, -negSum-posSum)))
          #prob=float(1/(1+posSum))
        if example.klass == 'neg': # this keeps updating theta vector or the weights vector
          y=0
          prob=float(1/(1+pow(2.718,posSum+negSum)))
          #prob=float(1/(1+negSum))
        print prob
        currentWeights=weights+(eta*((-lambdaa*weights)+(y-prob)*X))/m
        delta=findDelta(weights,currentWeights)
        count=count+1
        weights = currentWeights

  printResults()

def findDelta(weights,currentWeights):
  max=0
  for i in range(0,length):
    if np.abs(weights[i,0]-currentWeights[i,0])>max:
      max=np.abs(weights[i,0]-currentWeights[i,0])
  return max

#returns either 1 or 0 based on if feature is found or not

def findFeatureMatrixForDataSet(words,documentId):
  global XMap
  if documentId != None and documentId in XMap:
    return XMap[documentId]

  pos = 0
  val = 0
  X = np.zeros((length, 1))
  for word in words:
    word = stemmer.stem(word.lower())
    if word in features:
      i = features.index(word)
      X[i] = X[i] + 1  # here X is represented by total number of repetitions of the word
  XMap[documentId] = X
  return XMap[documentId]

def printResults():
  print "Weight Array "
  print weights.transpose()

def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)



if __name__ == "__main__":
    main()