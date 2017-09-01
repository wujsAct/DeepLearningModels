from spacy.en import English
import codecs
nlp = English()
#doc = nlp(u'''I want to go home and have supper.''')
#doc = nlp(u'''Perhaps this tragedy of historical proportions can become the foundation for a historic shift in human awareness.''')
#doc = nlp(u'''Dentists suggest that you should brush your teeth''')
#doc = nlp(u'Tom and Jerry are fighting!')
#strs=u'''what films did Patsha Bay act in'''
#strs = u'''what does John Ericson appear in''' 
#strs = u'''but instead there was a funeral , at st. fran
#cis de sales roman catholic church , in belle_harbor , queens , the parish of his birth'''
path ="data/kbp/LDC2017EDL/data/2014/training/source_documents/" +"bolt-eng-DF-170-181137-9033614.txt"
strs = codecs.open(path,'r','utf-8').read()
#strs = u'''gpri...@franklinprinting.com'''
print len(strs)
doc = nlp(strs)
i =0
allwords = 0
sentid= 0
for sentence in doc.sents:
  dep_triple = []
  senti=[]
  for token in sentence:
    #print token,'\t',token.tag_,'\t',token.pos_
    senti.append([token,token.idx,token.tag_,token.pos_])
  print senti
  sentid += 1
  for key in senti:
    allwords += len(key)
print allwords + sentid
#    temp = []
#    temp.append([token.orth_,t[token.idx]])
#    temp.append(nlp.vocab.strings[token.dep])
#    temp.append([token.head.orth_,t[token.head.idx]])
#    dep_triple.append(temp)
  
    
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
