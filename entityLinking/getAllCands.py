# -*- coding: utf-8 -*-
'''
@time: 2016/12/5
@editor: wujs
@function: to generate the final candidate
@revise: 2017/3/20 revise all the score! it may help us to score the results!
'''
import sys
reload(sys)
sys.path.append('utils')
sys.path.append('main1')
sys.path.append('main2')
sys.setdefaultencoding('utf8')

import cPickle
from wiki2fb import getWikititle2Freebase,getWikidata2Freebase,getCrossWiki
from reverseIndexUtils import getreverseIndex
from PhraseRecord import EntRecord
from tqdm import tqdm
import collections
from NGDUtils import NGDUtils

def getmid2Name():
  mid2name = collections.defaultdict(list) 
  #mid2name = {}
  '''
  @we ignore that a mid may have several names!
  '''
  with open('data/mid2name.tsv','r') as file:
    for line in tqdm(file):
      line =line.strip()
      #print line.split(u'\t')
      items= line.split('\t')
      
      if len(items)>=2:
        mid = items[0]; name = ' '.join(items[1:])
        mid2name[mid].append(name)
      else:
        items = line.split(' ')
        if len(items)>=2:
          mid = items[0]; name = ' '.join(items[1:])
          mid2name[mid].append(name)
        else:
          print line
  print len(mid2name)
  return mid2name

def is_contain_ents(enti,entj):
  enti = enti.lower()
  entj = entj.lower()
  if enti in entj:
    return True
  else:
    return False

def getMidIncomingLinks(ngd,mid2name,fmid):
  wmid_set = set()
  if fmid in mid2name:
    for title in mid2name[fmid]:
      wmid_set = wmid_set | ngd.getIncomingLinks(title)
  return wmid_set

def getMidAllLinks(ngd,mid2name,fmid):
  wmid_set = set()
  if fmid in mid2name:
    for title in mid2name[fmid]:
      pageContain,pageInclude = ngd.getLinkedEnts(title)
      wmid_set = wmid_set | set(pageContain.keys())
      wmid_set = wmid_set | set(pageInclude.keys())
  return wmid_set 
  

def get_freebase_ent_cands(mid2incomingLinks,ngd,mid2name,candent_mid2,enti,entstr2id,wikititle2fb,wikititle_reverse_index,freebaseNum):
  enti_title = enti.lower()
  totaldict=dict()
  '''
  @need to revise the scores
  '''
  if enti_title in wikititle_reverse_index:
    totaldict = wikititle_reverse_index[enti_title]
  elif enti_title.replace(' ','') in wikititle_reverse_index:
    totaldict = wikititle_reverse_index[enti_title.replace(' ','')]
  #print len(totaldict)

  freebase_cants = collections.defaultdict(int)
  ids_key=0
  for key in totaldict:
    if is_contain_ents(enti_title,key) or is_contain_ents(enti_title.replace(' ',''),key):
      addScore = 0
      
      if enti_title == key or enti_title.replace(' ','')== key or key in entstr2id:  #completely match, score is too low!
        addScore += 1
        
        for wmid in wikititle2fb[key]:  
          freebase_cants[wmid] = 1
                        
          ids_key += 1
    
          if wmid not in candent_mid2:
            tempScore = 0
            if wmid in mid2name:
              if wmid in mid2incomingLinks:
                tempScore = len(mid2incomingLinks[wmid])
              else:
                wmid_set = getMidIncomingLinks(ngd,mid2name,wmid)
                mid2incomingLinks[wmid] = wmid_set
                tempScore = len(wmid_set)
            candent_mid2[wmid] = list([tempScore,0,addScore])  #not appear in the crosswiki
          else:
            prior = list(candent_mid2[wmid])
            prior[2]= addScore
            candent_mid2[wmid]= prior                     
  return candent_mid2,freebase_cants

'''
@we should score as the number of inconming links!!! ==>reasonable
'''
def get_candent_mid(candent_mid1,mid2incomingLinks,ngd,listentcs,w2fb,wikititle2fb,mid2name):
  wiki_cands={}
  mid_index = 0.0
  for cent in listentcs:
    mid_index +=1.0
    if isinstance(cent,dict):
      ids = cent['ids']
      titles = cent['title'].lower()
    else:
      ids = cent
      titles = cent.lower()
      
    if titles in wikititle2fb:
      for wmid in wikititle2fb[titles]:
        tempScore = 0
        if wmid in mid2name:
          if wmid in mid2incomingLinks:
            wmid_set = mid2incomingLinks[wmid]
          else:
            wmid_set = getMidIncomingLinks(ngd,mid2name,wmid)
            mid2incomingLinks[wmid] = wmid_set
          tempScore = len(wmid_set)
        wiki_cands[wmid]=1
        if wmid in candent_mid1:
          prior = candent_mid1[wmid]
        else:
          prior = [0,0,0]
        prior[1] = tempScore
        candent_mid1[wmid] = list(prior)
  return candent_mid1,wiki_cands

   
'''
@generate the final candidate entity
1. wikipedia opensearch api
2. crosswiki alias
3. freebase alias
'''
def get_final_ent_cands():
  all_candidate_mids = []
  wiki_candidate_mids=[]
  freebase_candidate_mids=[]
  
  all_entment_numbers = 0

  ent_ment_link_tags = cPickle.load(open('data/aida/aida-annotation.p_new','rb'))
  if data_flag=='train':
    ent_id = 0
  if data_flag=='testa':
    ent_id = 23396
  if data_flag=='testb':
    ent_id = 29313
  print 'finish load all datas'
  
  #exit(-1)
  right_nums = 0;wrong_nums =0;pass_nums = 0
  totalRepCand = 0
  
  for i in tqdm(range(len(ent_Mentions))):
    aNosNo = id2aNosNo[i]
    docId = aNosNo.split('_')[0]
    ents = ent_Mentions[i]
    context_ents = docId_entstr2id[docId]
    for j in range(len(ents)):
      isRepflag =False
      
      all_entment_numbers+=1
      
      enti = ents[j]
      enti_name = enti.content.lower()
      candent_mid1 = {}
      #we add the crosswiki alias
      if enti_name in p_e_m:
        for key in p_e_m[enti_name]:
          candent_mid1[key] = [p_e_m[enti_name][key],0,0]
    
      startI = enti.startIndex; endI = enti.endIndex
      ment_key = aNosNo+'\t'+str(startI)+'\t'+str(endI)
     
      enti_linktag_item = ent_ment_link_tags[ent_id]
      tag = enti_linktag_item[1]
      
      if tag == 'NIL':
        pass_nums = pass_nums + 1
#        ent_id += 1
#        continue
      ent_id += 1
      '''
      we utilize the representative mention's candidate entity as the gold candidates 
      '''
      if ment_key in entMent2repMent:
        isRepflag = True
        enti_name = entMent2repMent[ment_key].split('\t')[-1].lower()
        
      entids = entstr2id[enti_name]
      
      
      listentcs = []
      for entid in entids:
        for entid_mid in candiate_ent[entid]:
          if entid_mid not in listentcs:
            listentcs.append(entid_mid)
            
      candent_mid2,wiki_cands = get_candent_mid(candent_mid1,mid2incomingLinks,ngd,listentcs,w2fb,wikititle2fb,mid2name)
      
      freebaseNum = max(0,90 - len(candent_mid1))
      final_mid,freebase_cands = get_freebase_ent_cands(mid2incomingLinks,ngd,mid2name,dict(candent_mid2),enti.content,context_ents,wikititle2fb,wikititle_reverse_index,freebaseNum)
      
      if tag in final_mid:
        if isRepflag:
          totalRepCand += 1
        right_nums += 1
      else:
        if tag != 'NIL':
          wrong_nums = wrong_nums + 1
        #print 'wrong:',tag,enti.content,totalCand#,final_mid
        #print candent_mid1,candent_mid2,candent_mid3
        #exit()
        
      all_candidate_mids.append(dict(final_mid))
      wiki_candidate_mids.append(dict(candent_mid2).keys())
      freebase_candidate_mids.append(dict(freebase_cands).keys())
  print '--------------'
  print 'data tag:',data_flag
  print 'wrong_nums:',wrong_nums
  print 'right_nums:',right_nums
  print 'pass_nums:',pass_nums
  print len(all_candidate_mids)
  print 'totalRep right:',totalRepCand
  return all_candidate_mids,wiki_candidate_mids,freebase_candidate_mids



if __name__=='__main__':
  if len(sys.argv) !=6:
    print 'usage: python pyfile dir_path inputfile train_entms.p100(test) train_ent_cand_mid.p flag'
    exit(1)
  dir_path = sys.argv[1]
  f_input = dir_path  +'/process/'+ sys.argv[2]
  f_input_entMents = dir_path  +'/features/'+ sys.argv[3]
  f_output = dir_path  +'/features/'+ sys.argv[4] #record the total entity nums.
  data_flag = sys.argv[5]
  #data context:  para_dict={'entstr2id':entstr2id,'candiate_ent':candiate_ent,'candiate_coCurrEnts':candiate_coCurrEnts}
  data = cPickle.load(open(f_input,'r'))
  entstr2id_org = data['entstr2id']
  #print entstr2id_org
  candiate_ent = data['candiate_ent']#;candiate_coCurrEnts = data['candiate_coCurrEnts']
  print 'entstr2id_org',len(entstr2id_org)
  id2entstr_org = {value:key for key,value in entstr2id_org.items()}
  entstr2id= collections.defaultdict(set)
  for key,value in entstr2id_org.items():
    entstr2id[key.lower()].add(value)
  print 'entstr2id',len(entstr2id)
  mid2name = getmid2Name()
  ngd = NGDUtils()
  w2fb = getWikidata2Freebase()
  wikititle2fb = getWikititle2Freebase(low_flag=True)  #不用lower
  
  entMent2repMent_org = cPickle.load(open(dir_path+'process/'+data_flag+'_entMent2repMent.p','rb'))  
  #print entMent2repMent_org
  
  print len(entMent2repMent_org)
  
  entMent2repMent = {}
  for key in entMent2repMent_org:
    keyr = '\t'.join(key.split('\t')[0:3])
    val = entMent2repMent_org[key].split('\t')[-1]
  
    entMent2repMent[keyr] = val
             
  data = cPickle.load(open(dir_path+'process/'+ data_flag+'.p','r'))
  
  aNosNo2id = data['aNosNo2id']
  id2aNosNo = {val:key for key,val in aNosNo2id.items()}
  
  
  #print candiate_ent
  #param_dict={'ent_Mentions':ent_Mentions,'aNo_has_ents':aNo_has_ents,'ent_ctxs':ent_ctxs} ==>
  dataEnts = cPickle.load(open(f_input_entMents,'r'))
  
  ent_Mentions = dataEnts['ent_Mentions']; aNo_has_ents=dataEnts['aNo_has_ents']#;ent_ctxs=dataEnts['ent_ctxs']
  print 'all entity mention numbers:',len(ent_Mentions)
 
  docId_entstr2id= collections.defaultdict(dict)
  '''
  @2017/3/1, we revise the entstr2id  to docid_entstr2id
  '''
  for i in tqdm(range(len(ent_Mentions))):
    aNosNo = id2aNosNo[i]
    docId = aNosNo.split('_')[0]
    ents = ent_Mentions[i]
    
    for j in range(len(ents)):
      enti = ents[j]
      enti_name = enti.content
      value = entstr2id_org.get(enti_name)
      if enti_name.lower() not in docId_entstr2id[docId]:
        docId_entstr2id[docId][enti_name.lower()]= {value}
      else:
        docId_entstr2id[docId][enti_name.lower()].add(value)
  #print docId_entstr2id
  
  print 'start to load wikititle...'
  wikititle_reverse_index  = getreverseIndex()
  p_e_m = getCrossWiki()
  print 'start to solve problems...'
  mid2incomingLinks={}
  mid2allLinks={}
  all_candidate_mids,wiki_candidate_mids,freebase_candidate_mids = get_final_ent_cands()
  data_param={'all_candidate_mids':all_candidate_mids,'wiki_candidate_mids':wiki_candidate_mids,'freebase_candidate_mids':freebase_candidate_mids}
  cPickle.dump(data_param,open(f_output,'wb'))
 