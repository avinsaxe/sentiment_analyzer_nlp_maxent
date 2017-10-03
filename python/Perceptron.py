import os
import numpy as np
import nltk
import nltk.classify.util
from nltk.stem.porter import PorterStemmer
np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)
from nltk.corpus import opinion_lexicon
nltk.download('opinion_lexicon')
global BOW
global negBOW
global posBOW
global features
global X
global weights
global positiveWords
global negativeWords
global XMap
global wordCount

#**********Section of Input Starts*********************
global maxWords
global eta
eta=1

#**********Section of Input Ends*********************


wordCount=0
BOW = set()
#these are maps of word with corresponding counts
BOW1 = {}
negBOW = {}
posBOW = {}

positiveWords={}
negativeWords={}
XMap={'A':'B'}  #Map of X for all the documents
stemmer = PorterStemmer()
weights=[]
positiveWords={}
negativeWords={}
negatives=opinion_lexicon.negative()
positives=opinion_lexicon.positive()
unit_step = lambda x: 0 if x < 0 else 1

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
    documentId=None
    X=findFeatureMatrixForDataSet(words,documentId)
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
    global maxWords
    global wordCount
    for r1 in words:
        r = stemmer.stem(r1)
        if klass == 'pos' and r in positives:
            if r not in posBOW:
                posBOW[r.lower()] = 1
            else:
                posBOW[r.lower()] = posBOW.get(r.lower()) + 1
            wordCount = wordCount + 1
        elif klass == 'neg' and r in negatives:
            if r not in negBOW:
                negBOW[r.lower()] = 1
            else:
                negBOW[r.lower()] = negBOW.get(r.lower()) + 1
            wordCount = wordCount + 1
    pass
  
  def train(self, split, iterations):
      documentId=1
      global eta
      global weights
      global length

      for i in range(0, iterations):
        documentId=1
        for example in split.train:
          words = example.words
          X=findFeatureMatrixForDataSet(words,documentId)
          documentId=documentId+1
          arr = np.dot(weights.transpose(),X)
          score=arr[0,0]
          y=0
          if (score>=0 and example.klass=='neg') or (score<0 and example.klass=='pos'):
            if example.klass == 'pos':
              y=1
            if example.klass == 'neg':
              y=0
            #print score
            expected=y
            result=score
            error = expected - unit_step(result)
            weights=weights+eta*error*X
        eta=eta/3

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

  def initializeBOW(self, splits):
        global BOW
        global weights
        global features
        global BOW1
        global posBOW
        global negBOW
        global maxWords
        global wordCount
        global negFeatures
        global posFeatures
        global length
        for w1 in positives:
            posBOW[w1] = 1
        for w1 in negatives:
            negBOW[w1] = 1
        wordCount = len(posBOW) + len(negBOW)
        #wordCount = len(negFeatures) + len(posFeatures)
        for split in splits:
            for example in split.train:
                if (maxWords > 0 and wordCount < maxWords):
                    words = example.words
                    self.addExample(example.klass, words)
                elif maxWords<0:
                    words = example.words
                    self.addExample(example.klass, words)
                elif maxWords>0 and wordCount>=maxWords:
                    break
        i = 0

        global posFeatures
        global negFeatures
        # TODO now we have to train our classifiers

        # print "Length is ",n
        features = []
        negFeatures = []
        posFeatures = []
        BOW = list(negBOW.keys())
        BOW = BOW + list(posBOW.keys())
        n = len(BOW)
        length = n + 1

        features = list(BOW)
        features.append('1')
        negFeatures = list(negBOW)
        posFeatures = list(posBOW)

        length = n + 1
        X = np.zeros(((n + 1),1))  # fills all X with 0s initially. Then start changing X based on input we are talking about in the document
        length = n + 1
        X = np.zeros((length,1))  # fills all X with 0s initially. Then start changing X based on input we are talking about in the document
        weights = np.zeros(((length), 1))
        for f in features:
            if f in posBOW and features.index(f)<len(weights):
                weights[features.index(f)] = posBOW.get(f)
            if f in negBOW and features.index(f)<len(weights):
                weights[features.index(f)] = -negBOW.get(f)

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



def test10Fold(iterations,maxWords1,eta1,trainDir):
  global maxWords
  global eta
  maxWords=maxWords1
  eta=eta1
  pt = Perceptron()
  splits = pt.crossValidationSplits(trainDir)
  avgAccuracy = 0.0
  fold = 0
  classifier = Perceptron()
  classifier.initializeBOW(splits)
  output=''
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
    output=output+'[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) +'\n'
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
  output=output+'[INFO]\tAccuracy: %f' % avgAccuracy+'\n'
  return output
    
def classifyDir(iter,maxWords1,eta1,trainDir1, testDir1):
  global maxWords
  global eta
  maxWords = maxWords1
  eta = eta1
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir1)
  iterations = int(iter)
  classifier.initializeBOW(trainSplit)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir1)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy
  return '[INFO]\tAccuracy: %f' % accuracy

def findFeatureMatrixForDataSet(words,documentId):
  global XMap
  global length

  if documentId!=None and documentId in XMap:
      return XMap[documentId]
  X=np.zeros((length,1))
  for word in words:
    word=stemmer.stem(word.lower())
    if word in features:
      i=features.index(word)
      if i<len(X):
        X[i]=X[i]+1  #here X is represented by total number of repetitions of the word
  XMap[documentId]=X
  return XMap[documentId]

def main(methodName,eta,maxWords,testDir,trainDir,iterations):
    if methodName=='test10Fold':
        return test10Fold(iterations,maxWords,eta,trainDir)
    elif methodName=='classifyDir':
        return classifyDir(iterations,maxWords,eta,trainDir,testDir)

if __name__ == "__main__":
    main('test10Fold',1, 1000, '../data/imdb1',100)
    #main('classifyDir',1,1000,'../data/imdb1','../data/imdb1',100)
