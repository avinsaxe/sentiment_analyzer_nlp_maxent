import sys
import getopt
import os

import MaxEnt
import re

def startFunctionTest():
    #methodName,epsilon,eta,maxCount,maxWords,testDir,trainDir
    inputs=[]
    results=[]
    inputFile='./Input/inputMaxent.txt'
    lines=[]

    with open(inputFile) as f:
        for line in f:
            arr=[]
            arr=line.split()
            inputs.append(arr)

    for input in inputs:
        if input[0]=='test10Fold':
            results.append(MaxEnt.main('test10Fold',float(input[1]),float(input[2]),float(input[3]),float(input[4]),input[5],input[6]))
        if input[0]=='classifyDir':
            results.append(MaxEnt.main('classifyDir',float(input[1]),float(input[2]),float(input[3]),float(input[4]),input[5],input[6]))
         #test10Fold takes only training directory and uses 10 fold to solve it
    #maxent.main('classifyDir', 0.1, 2, 50, 10000, '../data/imdb1', '../data/imdb1')

    createOutputFile('./Output/outputMaxent.txt',inputs,results)

def main():
  startFunctionTest()

def createOutputFile(filename,inputs,results):
    for i in range(0,len(results)):
        f = open(filename,'w')
        print >>f,inputs[i]
        print >>f,results[i]

if __name__ == "__main__":
    main()