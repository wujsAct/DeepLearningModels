# -*- coding: utf-8 -*-
import codecs
import gzip
import cPickle
import collections
from tqdm import tqdm


def un_gz2gettype(file_name):
  """ungz zip file"""
  #创建gzip对象
  
  if 'gz' in file_name:
    fileobj = gzip.open(file_name)
  else:
    fileobj = codecs.open(file_name)
  return fileobj
    
def getmid2type(fileobj,output_file):
  reltag = '<http://rdf.freebase.com/ns/type.object.type>'
  enttag = '<http://rdf.freebase.com/ns/m.'
  result = codecs.open(output_file,'w','utf-8')
  linenum =0
  while 1:
    lines = fileobj.readlines(100000)
    if not lines:
      break
    for line in lines:
      line = line.strip()
      items = line.split('\t')
      if len(items)>=4:
        ent1 = items[0];rel = items[1]
        if (enttag in ent1) and (reltag in rel):
          ent1 = '/m/'+ent1.split('<http://rdf.freebase.com/ns/m.')[1].replace('>','')  #mid形式
          typei = items[2]
          typei = typei.split('ns/')[-1]
          typei = typei.replace('.','/')
          typei = '/'+typei.replace('>','')
          result.write(ent1+'\t'+typei+'\n')
          linenum = linenum+ 1
          if linenum %1000000==0:
            print linenum
  
  result.close()

def getmid2description(file_name,output_file):
  reltag = '<http://rdf.freebase.com/ns/common.topic.description>'
  enttag = '<http://rdf.freebase.com/ns/m.'
  result = codecs.open(output_file,'w','utf-8')
  linenum =0
  with codecs.open(file_name,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split('\t')
      if len(items)>=4:
        ent1 = items[0]; rel = items[1]
        if (enttag in ent1) and (reltag in rel):
          ent1 = u'/m/'+ent1.split(u'<http://rdf.freebase.com/ns/m.')[1].replace(u'>',u'')  #mid形式
          description = items[2]
          if description.endswith(u'@en'):
            result.write(ent1+u'\t'+description+u'\n')
      linenum += 1
      if linenum % 1000000==0:
        print linenum
  
  result.close()

def getfreebasetype2figertype(file_name,dir_path):
  figer_dict={}
  freebasetype2figertype={}
  figer_type_id = 0
  with codecs.open(file_name,'r','utf-8') as file:
    for line in file:
      line = line.strip()
      items = line.split(u"\t")
      freebasetype =items[0]; figertype = items[1]
      freebasetype2figertype[freebasetype] = figertype
      if figertype not in figer_dict:
        figer_dict[figertype] = figer_type_id
        figer_type_id += 1
  param_dict={'figer_dict':figer_dict,'freebasetype2figertype':freebasetype2figertype}
  cPickle.dump(param_dict,open(dir_path+'free2figer.p','wb'))
  
  print 'figer total types:',len(figer_dict), figer_type_id
  print figer_dict   

def getmid2figertype(file_name,dir_path):
  #param_dict={'figer_dict':figer_dict,'freebasetype2figertype':freebasetype2figertype}
  param_dict = cPickle.load(open(dir_path+'free2figer.p','rb'))
  figer_dict = param_dict['figer_dict']; freebasetype2figertype = param_dict['freebasetype2figertype']
  
  mid2figer= collections.defaultdict(list)
  totalmid=set()
  with codecs.open(file_name,'r','utf-8') as file:
    for line in tqdm(file):
      line = line.strip()
      items = line.split(u"\t")
      mid = items[0]; freebasetype = items[1]
      totalmid.add(mid)
      if freebasetype in freebasetype2figertype:
        mid2figer[mid].append(figer_dict[freebasetype2figertype[freebasetype]])
  cPickle.dump(mid2figer,open(dir_path+'mid2figer.p','wb'))      
  print 'totalmid:',len(totalmid), 'mid2figer:',len(mid2figer)
dir_path = '/home/wjs/demo/entityType/informationExtract/data/'
fname = dir_path+'freebase-rdf-latest'
#g_file_fb = un_gz2gettype(fname)
#output_file = dir_path+'mid2type.txt'
#getmid2type(g_file_fb,output_file)
#output_file = dir_path+'mid2description.txt'
#getmid2description(fname,output_file)

#file_name = dir_path+'freebasetype2figertype.map'
#getfreebasetype2figertype(file_name,dir_path)

file_name = dir_path + 'mid2type.txt'
getmid2figertype(file_name,dir_path)