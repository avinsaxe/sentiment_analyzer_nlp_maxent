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

BOW = set()
#these are maps of word with corresponding counts
BOW1 = {}
negBOW = {}
posBOW = {}
maxWords=20000
positiveWords={}
negativeWords={}

stemmer = PorterStemmer()

class Perceptron:
  stopwords = nltk.corpus.stopwords.words('english')
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
    """Perceptron initialization"""
    self.numFolds = 10

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Perceptron classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    X=findFeatureMatrixForDataSet(words)
    arr=np.dot(X.transpose(),weights)
    score=arr[0,0]
    if score>=0:
      return 'pos'
    return 'neg'
    # Write code here


  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
    """

    global BOW
    global weights
    global features
    global BOW1
    global posBOW
    global negBOW
    posBOW={}
    negBOW={}
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

    pass
  
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights

      """
      global BOW
      global weights
      global features
      global BOW1
      global posBOW
      global negBOW

      for example in split.train:
        words = example.words
        self.addExample(example.klass, words)



      i = 0

      posBOW = sorted(posBOW, key=posBOW.get,
                      reverse=True)  # sort the map of BOW1 based on value of each key in descending order
      negBOW = sorted(negBOW, key=negBOW.get,
                      reverse=True)  # sort the map of BOW1 based on value of each key in descending order
      list1 = posBOW[:maxWords / 2] + list(negBOW[:maxWords / 2])
      BOW = set(list1)
      global posFeatures
      global negFeatures
      # TODO now we have to train our classifiers
      n = len(BOW)
      # print "Length is ",n
      features = []
      negFeatures = []
      posFeatures = []

      features = list(BOW)
      features.append('1')
      negFeatures = list(negBOW)
      posFeatures = list(posBOW)

      global length
      length = n + 1
      X = np.zeros(((n + 1),
                    1))  # fills all X with 0s initially. Then start changing X based on input we are talking about in the document
      weights = np.zeros(((n + 1), 1))
      weights = np.random.rand(n + 1, 1)
      for i in range(0,n):
       r=features[i]
       weights[i,0] = -np.random.random()


      for i in range(0, iterations):
        for example in split.train:
          words = example.words
          X=findFeatureMatrixForDataSet(words)
          arr = np.dot(weights.transpose(),X)
          score=arr[0,0]
          y=0
          if (score>=0 and example.klass=='pos') or (score<0 and example.klass=='neg'):
            if example.klass == 'pos':
              y=1
            if example.klass == 'neg':
              y=0
            weights=weights+(y-score)*X



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

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
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
  pt = Perceptron()
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)

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
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy

def findFeatureMatrixForDataSet(words):
  pos=0
  val=0
  X=np.zeros((length,1))
  for word in words:
    word=stemmer.stem(word.lower())
    if word in features:
      i=features.index(word)
      X[i]=X[i]+1  #here X is represented by total number of repetitions of the word
  return X

def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
