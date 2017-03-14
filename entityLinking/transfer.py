# -*- coding: utf-8 -*-

import sys
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')

import codecs
from spacyUtils import spacyUtils
from PhraseRecord import EntRecord
from spacy.en import English
import cPickle

def extractEntMention(mentag,sentence_list):
  
    
  #utilzie two pointers to solve the problem, need to improve the algorithms ability!
  lent = len(mentag)
  entMen = []
  entType= []
  p = 0;q = 0
  while(q<lent and p < lent):
    if mentag[p]==U'O':
      p = p + 1
      q = p
    else:
      if mentag[q]==U'O':
        entName = u' '.join(sentence_list[p:q])
        if entName.lower() == u'county':
          print 'county:',len(sentence_list)
          exit()
        temp = EntRecord(p,q)
        temp.setContent(sentence_list)
        entMen.append(temp)
        entType.append(mentag[p])
        #print u' '.join(sentence_list[p:q]),'\t',mentag[p]
        p=q+1
        q=p
      else:
        if (mentag[q] == mentag[p] and (p==q or mentag[q].split('-')[0]!=u'B')) or (mentag[q].split('-')[1] == mentag[p].split('-')[1] and mentag[q].split('-')[0]!=u'B'): 
          q = q + 1
          if q == lent:
            #print u' '.join(sentence_list[p:q]),'\t',mentag[p]
            entName = u' '.join(sentence_list[p:q])
            if entName.lower() == 'county':
              print sentence_list
              exit()
            temp = EntRecord(p,q)
            temp.setContent(sentence_list)
            entMen.append(temp)
            entType.append(mentag[p])
            break
        else:
          entName = u' '.join(sentence_list[p:q])
          if entName.lower() == 'county':
            print sentence_list
            exit()
          temp = EntRecord(p,q)
          temp.setContent(sentence_list)
          entMen.append(temp)
          entType.append(mentag[p])
          #print u' '.join(sentence_list[p:q]),'\t',mentag[p]
          p = q
  return entMen,entType
      
      


if __name__=='__main__':
  if len(sys.argv) !=7:
    print 'usage: python pyfile dir_path inputfile outputfile'
    exit(1)
  
  dir_path = sys.argv[1]
  f_input = dir_path + sys.argv[2]
  f_output = dir_path +'process/' + sys.argv[3]
  f_sents = codecs.open(dir_path +'features/' + sys.argv[4],'w','utf-8')
  f_sentid2aNosNo = codecs.open(dir_path +'features/' + sys.argv[5],'w','utf-8')
  f_ent2aNosNo = dir_path +'features/' + sys.argv[6]
  
  '''
  @revise 2017/3/1 '''
  
  #nlp = English() #need to feed in unicode sentence!
  maxLength = 0
  sentence_list=[]; postag=[];mentag =[]
  aNosNo2id = {};id2aNosNo={};aNosNoId = 0;ents=[];sents=[]
  ent2aNosNo = []
  all_sentence_list=[]  #to train word vector
  aNosNo2id={}
  totalEnts = 0
  with codecs.open(f_input,'r','utf-8') as file:
    allcontent = file.read().strip()
    allDocs = allcontent.split(u'-DOCSTART- -X- -X- O')
    print len(allDocs)
    
    sentid = 0
    for article_No in range(1,len(allDocs)):
      aNo = article_No - 1
      doc = allDocs[article_No].strip()
      
      
      sent_doc = doc.split(u'\n\n')
      for sentence_No in range(len(sent_doc)):
        sent_ents = []
        senti = sent_doc[sentence_No]
        sentiLent = len(senti.split(u'\n'))
        if sentiLent > maxLength:
          maxLength = sentiLent
        sentence_list=[]; mentag =[]
        aNosNo_str = str(aNo)+'_'+str(sentence_No)
        aNosNo2id[aNosNo_str] = sentid
        id2aNosNo[sentid] = aNosNo_str
        f_sentid2aNosNo.write(aNosNo_str+'\n')
        #print aNosNo_str
        sentid += 1
                 
        for line in senti.split(u'\n'):
          f_sents.write(line+'\n')
          item = line.split(u' ')
          sentence_list.append(item[0])
          mentag.append(item[3])
        f_sents.write('\n')
        sents.append(u' '.join(sentence_list))
        entMen,entType = extractEntMention(mentag,sentence_list)
        totalEnts += len(entMen)
        for key in entMen:
          ent2aNosNo.append(aNosNo_str)
        ents.append([entMen,entType])
      
    assert len(aNosNo2id) == len(ents)
    assert len(ents)==len(sents)
    assert len(ent2aNosNo) == totalEnts
      
    '''
    for line in file:
      line = line[:-1]
      #start of the article
      if line == '-DOCSTART- -X- -X- O':
        article_No += 1
        sentence_No=-1
      else:
        if line != u'':
          item = line.split(u' ')
          sentence_list.append(item[0])
          mentag.append(item[3])
        else:
          #start an new sentence
          if len(sentence_list)!=0:
            sentence_No = sentence_No+1
            sentence = u' '.join(sentence_list)
            all_sentence_list.append(sentence)
            sputils=spacyUtils(sentence,nlp)
            tag = sputils.getPosTags()
            depTree = sputils.getDepTree()
            #print sentence
            entMen,entType = extractEntMention(mentag,sentence_list)
            entlents = 0
            for enti in entMen:
              entlents = entlents + len(enti.content.split(u' '))
            nonOlent = 0
            for menti in mentag:
              if menti!=u'O':
                nonOlent = nonOlent + 1
            assert entlents == nonOlent
            if len(entMen):
              #sentence_No = sentence_No+1
              #print sentence_list
              aNosNo_str = str(article_No)+'_'+str(sentence_No)
              print aNosNo_str
              if tag is None:
                print 'tag is None'
                exit(-1)
              if aNosNo_str in aNosNo2id:
                print 'dict_aNosNo is already in the dict!'
              
              aNosNo2id[aNosNo_str] = aNosNoId
              id2aNosNo[aNosNoId] = aNosNo_str
              aNosNoId = aNosNoId + 1
              ents.append([entMen,entType])
              tags.append(tag)
              sents.append(sentence)
              depTrees.append(depTree)
              mentags.append(mentag)
            sentence_list=[]; postag=[];mentag =[]
    '''
  #para_dict={'aNosNo2id':aNosNo2id,'id2aNosNo':id2aNosNo,'sents':sents,'tags':tags,'ents':ents,'mentags':mentags,
               #'depTrees':depTrees,'all_sentence_list':all_sentence_list}
  cPickle.dump(ent2aNosNo,open(f_ent2aNosNo,'wb'))
  
  print maxLength
  print 'start to save data...'
  para_dict ={'aNosNo2id':aNosNo2id,'id2aNosNo':id2aNosNo,'sents':sents,'ents':ents}
  cPickle.dump(para_dict,open(f_output,'wb'))
  f_sents.close()
  f_sentid2aNosNo.close()
#  print 
#  allents =0
#  for key in ents:
#    allents+= len(key[0])
#  print allents
#  cPickle.dump(ents,open(f_output1,'wb'))
 
#  
