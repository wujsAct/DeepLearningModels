from collections import defaultdict

doc_a={'id':'a','words':['word_w','word_x','word_y']}
doc_b={'id':'b','words':['word_x','word_z']}
doc_c={'id':'c','words':['word_y']}

docs =[doc_a,doc_b,doc_c]
indices = defaultdict(list)

for doc in docs:
  for word in doc['words']:
    indices[word].append(doc['id'])

print indices

def word_count(words):
  count =defaultdict(int)
  for word in words:
    count[word] += 1
  return count

print word_count(['hirrr'])
