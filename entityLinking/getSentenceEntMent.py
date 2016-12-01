#get aritcle_no sentence_no sentence entity_mentions

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
import multiprocessing
import urllib2
import codecs
import cPickle
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord
from urllibUtils import urllibUtils

  
if __name__=='__main__':
  if len(sys.argv) !=4:
    print 'usage: python pyfile dir_path inputfile outputfile'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  + sys.argv[2]
  f_output = dir_path + sys.argv[3]
  #f_input context is: para_dict={'aNosNo2id':aNosNo2id,'id2aNosNo':id2aNosNo,'sents':sents,'tags':tags,'ents':ents,'depTrees':depTrees}
  data = cPickle.load(open(f_input,'r'))
  aNosNo2id = data['aNosNo2id']; id2aNosNo=data['id2aNosNo']; sents=data['all_sentence_list']; ents=data['ents'];sents_real=data['sents'];tags=data['tags'];mentags=data['mentags']
  f_output_w = codecs.open(f_output,'w','utf-8')
  for i in xrange(len(sents)):
    f_output_w.write(sents[i]+'\n')
  f_output_w.close() 
  
