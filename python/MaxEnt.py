import os
import numpy as np
import nltk
import nltk.classify.util
from nltk.stem.porter import PorterStemmer
np.set_printoptions(threshold='nan')
from nltk.corpus import opinion_lexicon
nltk.download('opinion_lexicon')




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
global wordCount


#**********Section of Input Starts*********************
global maxWords
global epsilon
global eta
global lambdaa
global maxCount
epsilon=0.001
eta=0.01
lambdaa=0.1
maxCount=-1
maxWords=int(-1)

#**********Section of Input Ends*********************

wordCount=0
XMap={'A':'B'}
BOW = set()
#these are maps of word with corresponding counts
BOW1 = {}
negBOW = {}
posBOW = {}
positiveWords={}
negativeWords={}
negatives=opinion_lexicon.negative()
positives=opinion_lexicon.positive()
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
    global wordCount
    for r1 in words:
            r=stemmer.stem(r1)
            #print "Word Count ******* ", wordCount
            if klass=='pos' and r in positives:
              if r not in posBOW:
                posBOW[r.lower()]=1
              else:
                posBOW[r.lower()]=posBOW[r.lower()]+1
              #print "Positive ",r.lower(), " ", posBOW[r.lower()]
              wordCount=wordCount+1
            elif klass =='neg' and r in negatives:
              if r not in negBOW:
                negBOW[r.lower()]=1
              else:
                negBOW[r.lower()]=negBOW[r.lower()]+1
              #print "Negative ", r.lower(), " ", negBOW[r.lower()]
              wordCount = wordCount + 1
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
      global wordCount
      global length

      posBOW={}
      negBOW={}
      wordCount=0
      for w1 in positives:
        posBOW[w1]=1
      for w1 in negatives:
        negBOW[w1] = 1
      wordCount=len(posBOW)+len(negBOW)

      for split in splits:
        for example in split.train:
          if (maxWords > 0 and wordCount < maxWords):
            words = example.words
            self.addExample(example.klass, words)
          elif maxWords < 0:
            words = example.words
            self.addExample(example.klass, words)
          elif maxWords > 0 and wordCount >= maxWords:
            break

      i=0


      BOW=list(negBOW.keys())
      BOW=BOW+list(posBOW.keys())
#      BOW=set(posBOWList+negBOWList)
      global posFeatures
      global negFeatures
      #TODO now we have to train our classifiers
      n=len(BOW)
      features = []
      negFeatures=[]
      posFeatures=[]

      features=list(BOW)
      features.append('1')
      negFeatures=list(negBOW)
      posFeatures=list(posBOW)

      length=n+1
      X = np.zeros((n+1,1))   #fills all X with 0s initially. Then start changing X based on input we are talking about in the document
      weights = np.zeros((n+1,1))

      for f in features:
        if f in posBOW:
          weights[features.index(f)]=posBOW.get(f)
        if f in negBOW:
          weights[features.index(f)]=-negBOW.get(f)


     # weights=np.random.rand(n+1,1)/10

    #  for i in range(n/2,n+1):
     #      weights[i]=-weights[i]

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
  




def test10Fold(epsilon1,eta1,trainDir1,testDir1,maxWords1,maxCount1):
  global epsilon,eta,trainDir,testDir,maxWords,maxCount
  epsilon=epsilon1
  eta=eta1
  trainDir=trainDir1
  testDir=testDir1
  maxWords=maxWords1
  maxCount=maxCount1
  pt = Maxent()
  splits = pt.crossValidationSplits(trainDir)
  avgAccuracy = 0.0
  fold = 0
  classifier = Maxent()
  classifier.train(splits, epsilon, eta, lambdaa)

  # print "Weights of Features"
  # for i in range(0,len(weights)-1):
  #   print features[i]," =  ",weights[i]

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


def classifyDir(epsilon1, eta1, trainDir1, testDir1, maxWords1, maxCount1):
  global epsilon, eta, delta, trainDir, testDir, maxWords, maxCount
  epsilon = epsilon1
  eta = eta1

  trainDir = trainDir1
  testDir = testDir1
  maxWords = maxWords1
  maxCount = maxCount1
  pt = Maxent()
  splits = pt.crossValidationSplits(trainDir)
  classifier = Maxent()

  classifier.train(splits, epsilon, eta, lambdaa)  #TODO no training has been defined yet

  testSplit = classifier.trainSplit(testDir)  #again repeating same thing for testing directory
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
  global wordCount
  delta=1000000
  global maxCount
  count=0
  wordCount=len(posBOW)+len(negBOW)
  #print "Weights ", weights
  #print "features ",features
  #while delta>epsilon:
  print "*********gradient descent started***********"
  while delta>epsilon and (count <maxCount or maxCount<0) :
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
        #print prob
        currentWeights=weights+(eta*((-lambdaa*weights)+(y-prob)*X))/m
        delta=findDelta(weights,currentWeights)
        count=count+1
        weights = currentWeights

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

def main(methodName,epsilon,eta,maxCount,maxWords,testDir,trainDir):

  if methodName=='classifyDir':
    classifyDir(epsilon,eta,trainDir,testDir,maxWords,maxCount)
  elif methodName=='test10Fold':
    test10Fold(epsilon,eta,trainDir,testDir,maxWords,maxCount)


if __name__ == "__main__":
    #main('test10Fold', 0.1, 2, 50, 1000, '../data/imdb1') #test10Fold takes only training directory and uses 10 fold to solve it
    main('classifyDir',0.1,2,50,10000,'../data/imdb1','../data/imdb1')