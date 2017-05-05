from spacy.en import English

nlp = English()
#doc = nlp(u'''I want to go home and have supper.''')
#doc = nlp(u'''Perhaps this tragedy of historical proportions can become the foundation for a historic shift in human awareness.''')
#doc = nlp(u'''Dentists suggest that you should brush your teeth''')
#doc = nlp(u'Tom and Jerry are fighting!')
#strs=u'''what films did Patsha Bay act in'''
#strs = u'''what does John Ericson appear in''' 
#strs = u'''but instead there was a funeral , at st. fran
#cis de sales roman catholic church , in belle_harbor , queens , the parish of his birth'''
strs = u'''The band also shared membership with the similar , defunct group Out HUb ( including Tyler Pope, who has played with LCD Soundsystem and written music for Cake ) .'''

doc = nlp(strs)
i =0

for sentence in doc.sents:
  t = {token.idx:i for i,token in enumerate(sentence)}
  dep_triple = []
  for token in sentence:
#    if token.pos_ !='PUCNT':
    print token,'\t',token.tag_,'\t',token.pos_
    temp = []
    temp.append([token.orth_,t[token.idx]])
    temp.append(nlp.vocab.strings[token.dep])
    temp.append([token.head.orth_,t[token.head.idx]])
    dep_triple.append(temp)
    print temp

#
###use the knn to find the nearest neighbors
##from sklearn.neighbors import NearestNeighbors
##import numpy as np
##
##X = np.loadtxt('relationclustering/latent_relation.txt')
##nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
##distances, indices = nbrs.kneighbors(X)
##print indices[0]
##print distances[0]
