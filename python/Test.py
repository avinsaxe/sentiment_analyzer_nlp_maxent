import sys
import getopt
import os
import numpy as np
import math
import operator
import nltk
import re
import nltk
from nltk.corpus import wordnet as wn


a=set(['A','B','A','B','B'])
print a
a.add('C')
print a

print len(a)
a=set((10,2,2,3,41,5))
print a
a= sorted(a, reverse=True)



words = ['amazing', 'interesting', 'love', 'great', 'nice']

for w in words:
    tmp = wn.synsets(w)[0].pos()
    print w, ":", tmp



print a