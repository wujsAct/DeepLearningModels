# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:36:51 2017

@author: DELL
"""

import MySQLdb
db = MySQLdb.connect("10.1.1.22","root","123456","wiki" )
cursor = db.cursor()

sql = "select * from pagelinks limit 20"

try:
  #execute the sql command
  cursor.execute(sql)
  
  #fetch all the rows in a list of lists
  results = cursor.fetchall()
  for row in results:
    print row
except:
  print 'Error: unable to fetch data'
  
db.close()