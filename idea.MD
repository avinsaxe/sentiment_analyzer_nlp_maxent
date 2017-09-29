1. Each word in all the documents are a feature
  2. Associate a weight for all the features
  3. Try to change weight using Gradient Descent algorithm to associate a weight to all the features
  4. Compute the actual probability of a document to occur in a class c using the weights and probability function

  5. BOW=set.difference(set.union(posBOW,negBOW),set.intersection(posBOW,negBOW))
     Important. We are only considering words that are unique to either classification
            # here we took negBOW union posBOW - negBOW intersection posBOW

  6. We will further give slight negative weights to words occuring in negative set, and positive random values to elements in positive set

  7. features=np.matrix()
     X=np.matrix
     weights=np.matrix