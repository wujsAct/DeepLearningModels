# -*- coding: utf-8 -*-
"""
Created on Fri May 26 21:46:01 2017

@author: wujs
@function: build python ner server
"""
from BaseHTTPServer import BaseHTTPRequestHandler
from entityRecog import namedEntityRecognition
import re
try:
    from urllib.parse import urlparse
except ImportError:
     from urlparse import urlparse
import json
from urllib import unquote

def getNER(sents,pred):
  entMents = []
  sentNums = len(sents)
  classType = r'01*'   #greed matching, find the longest substring.
  pattern = re.compile(classType)
  entID= 1
  for i in range(sentNums):
    strs = ''.join(map(str,map(int,pred[i])))
    
    matchList = re.finditer(pattern,strs)  #very efficient layers!
    for match in matchList:
      s = match.start(); e = match.end()
      entMents.append({'Id':entID,'Mention':sents[i][s:e],'SentNo':i+1,'MentionIndex':str(s)+'_'+str(e)})
      entID += 1
  return entMents

class NERHandler(BaseHTTPRequestHandler):
   
  
  def do_GET(self):
    o = urlparse(unquote(self.path))
    rawData = json.loads(o.query)
    print rawData
    
    if o.path!='/NER':
      self.send_error(404,'File not found.')
      return
    pred = NER(rawData)  #call the ner function....
    
    entMents = getNER(rawData['tokenArrays'],pred)
    
    message = json.dumps(entMents)
    
    print entMents
    
    self.send_response(200)
    self.send_header('Content-type','application/json')
    self.end_headers()
    self.wfile.write(message)
    
if __name__=="__main__":
  from BaseHTTPServer import HTTPServer
  NER = namedEntityRecognition() 
  server = HTTPServer(('0.0.0.0',8000),NERHandler)
  print('Straring server, use <ctrl-c> to stop')
  server.serve_forever()
  
#PORT = 8000
#Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
#httpd = SocketServer.TCPServer(("",PORT),Handler)
#
#print "servring at port:",PORT
#
#httpd.serve_forever()