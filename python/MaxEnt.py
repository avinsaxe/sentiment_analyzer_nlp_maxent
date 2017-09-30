import sys
import getopt
import os
from fractions import Fraction

import numpy as np
import math
import operator
import nltk
import re
np.set_printoptions(threshold='nan')

global BOW
global maxWords
global negBOW
global posBOW
global features
global X
global weights

BOW = set()
BOW1 = set()
negBOW = set(['A', 'A'])
posBOW = set(['A', 'A'])
maxWords=20000


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

    X= findFeatureMatrixForDataSet(words)
    probPos=summation(1,X)
    probNeg=summation(0,X)
    print "probPos ",probPos
    print "probNeg ",probNeg
    if(probPos>probNeg):
      return 'pos'
    return 'neg'

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

    for r in words:
       if r not in self.stopwords:
         if re.match('\w',r) and len(r)>1:
            if klass=='pos':
                posBOW.add(r)
            elif klass =='neg':
                negBOW.add(r)



    # for r in self.BOW:
    #   print r

    # for r in words:
    # # text = self.pat/tern.sub('', r)

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
      for split in splits:
        for example in split.train:
            words = example.words
            if len(BOW1)<maxWords:
              self.addExample(example.klass, words)
              #BOW=set.union(posBOW,negBOW)
              BOW1=set.difference(set.union(posBOW,negBOW),set.intersection(posBOW,negBOW))
              # here we took negBOW union posBOW - negBOW intersection posBOW
            else:
              break
        #here we will have BOW initialized
      i=0
      for word1 in BOW1:
        if i<maxWords:
          BOW.add(word1)
          i=i+1
        else:
          break
      global posFeatures
      global negFeatures
      #TODO now we have to train our classifiers
      n=len(BOW)
      #print "Length is ",n
      BOW.add('A')  #added 1 as a represetation of [f0]  {f0,f1,f2,f3,...f}
      features = []
      negFeatures=[]
      posFeatures=[]

      for elem in list(BOW):
        features.append(elem)
      for elem in list(negBOW):
        negFeatures.append(elem)
      for elem in list(posBOW):
        posFeatures.append(elem)

      global length
      length=n+1
      X = np.zeros(((n+1),1))   #fills all X with 0s initially. Then start changing X based on input we are talking about in the document
      weights = np.zeros(((n+1),1))


      for i in range(0,n):
        r=features[i]
        val = np.random.choice(5, 1, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        if r in negFeatures:
          weights[i,0] = -np.random.random()
  #np.random.shuffle([0.3,0.4,0.1,0.2,0.7,0.6]);
        if r in posFeatures:
          weights[i,0] = np.random.random()
          #np.random.shuffle([0.3,0.4,0.1,0.2,0.7,0.6]);  #np.random.shuffle([-1,-2,-3,-1.5,-2.5,-3.5]);

      #Ideally gradient descent should give me appropriate weights, so that we can test the test set on it
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
  eta=1
  lambdaa=10
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  classifier = Maxent()
  classifier.train(splits, epsilon, eta, lambdaa)

  #print "BOW length ", len(BOW)
  #print BOW
  #print weights.transpose()
  #print features


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
    #print X.transpose()
    sum1=0.000000
    for i in range(0,len(X)):
      feature=features[i]  #we fetch the feature
      arr=np.array((weights[i] * X[i]))[0].tolist()
      #print arr
      if y==1 and feature in posFeatures:
        #val=np.power(2.718,arr[0])  # e^{lambda_i * f_i}
        sum1=sum1+arr
      if y==0 and feature in negFeatures:
        #val=np.power(2.718,arr[0])
        sum1=sum1+arr
    if product>0:
      return float(sum1)
    return 0.0

def gradientDescent(splits,epsilon,eta,lambdaa):
  global weights
  delta=1000000
  maxCount=30
  count=1

  #while delta>epsilon:
  print "*********gradient descent started***********"
  while delta>epsilon and count <maxCount :
    #print "Iteration ",delta
    i=0
    for split in splits:
      #print "iteration ",count
      #print "Slit ",i
      i=i+1

      m=len(split.train)
      sumMatFinal=np.zeros(((length),1))
      currentWeights=np.zeros(((length),1))
      for example in split.train:
        if count>maxCount:
          break
        words = example.words  #each document
        X = findFeatureMatrixForDataSet(words)
        factor=np.ones((length,1))
        #X=X+factor
        posSum=summation(1,X)
        negSum=summation(0,X)
        dif = negSum - posSum
        posSum = pow(2.718, dif)
        negSum=pow(2.718,-dif)
        prob=1.00000
        if example.klass == 'pos':
          y=1
          prob=float(1/(1+posSum))
          #print "Probability positive ",prob
        if example.klass == 'neg': # this keeps updating theta vector or the weights vector
          y=0
          prob=float(1/(1+negSum))
          #print "Probability negative ", prob
        currentWeights=weights+(eta*(y-prob)*X)/m
        delta=findDelta(weights,currentWeights)
        count=count+1
        weights = currentWeights

def findDelta(weights,currentWeights):
  max=0
  for i in range(0,length):
    if np.abs(weights[i,0]-currentWeights[i,0])>max:
      max=np.abs(weights[i,0]-currentWeights[i,0])
  return max

def findFeatureMatrixForDataSet(words):
  X=np.zeros(((length),1))
  pos=0
  for i in range(0,len(features)):
    word=features[i]
    X[pos,0]=words.count(word)
    pos=pos+1
  return X

def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)



if __name__ == "__main__":
    main()