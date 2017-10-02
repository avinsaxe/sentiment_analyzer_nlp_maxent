import sys
import getopt
import os

import MaxEnt

def startFunctionTest():
    #methodName,epsilon,eta,maxCount,maxWords,testDir,trainDir
    results=[]
    inputs=[]

    inputs.append(('test10Fold', 0.1, 2, 1, 10, 10,'../data/imdb1','../data/imdb1/'))
    results.append(MaxEnt.main('test10Fold', 0.1, 2, 10, 10, '../data/imdb1','../data/imdb1/')) #test10Fold takes only training directory and uses 10 fold to solve it
    #maxent.main('classifyDir', 0.1, 2, 50, 10000, '../data/imdb1', '../data/imdb1')

    createOutputFile('./Output/output.txt',inputs,results)

def main():
  startFunctionTest()

def createOutputFile(filename,inputs,results):
    for i in xrange(results):
        f = open(filename,'w')
        print >>f,inputs[i]
        print >>f,results[i]




if __name__ == "__main__":
    main()