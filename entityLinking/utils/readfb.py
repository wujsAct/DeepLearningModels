# -*- coding: utf-8 -*-
import codecs
import gzip

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
dir_path = '/home/wjs/demo/entityType/informationExtract/data/'
g_file_fb = un_gz2gettype(dir_path+'freebase-rdf-latest')
output_file = dir_path+'mid2type.txt'
getmid2type(g_file_fb,output_file)