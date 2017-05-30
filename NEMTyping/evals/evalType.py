import numpy as np

def getTypeEval(threshold,predArray,targetArray):
  right = 0; alls = 0
  '''
  @revise the accuracy test method
  '''
  precision=[]
  recall=[]
  
  right_cross=[]
  predRet_list = []
  targetRet_list=[]
  
  for i in range(len(predArray)):
    pred = predArray[i]
    target = targetArray[i]
    
    target_lents = len(np.nonzero(target)[0])*(-1)
    
    '''
    @top one must add in
    '''
    predRet=set()
    
    predRet.add(np.argmax(pred))
    
    for j in range(len(pred)):
      if pred[j] > threshold:   #fixed threshold is very diffcult to get! we just choose the 0.3
        predRet.add(j)
    
    targetRet = set(np.argsort(target)[target_lents:])
    
    rightset = predRet & targetRet
    right_cross.append(len(rightset))
    predRet_list.append(len(predRet))
    targetRet_list.append(len(targetRet))
    
    if len(rightset)==0:
      precision.append(0.0)
    else:
      precision.append(len(rightset)*1.0/len(predRet))
    
    recall.append(len(rightset) *1.0/len(targetRet))
  
    if predRet == targetRet:
      right += 1
    alls += 1
  return right,alls,precision,recall,right_cross,predRet_list,targetRet_list


def getRelTypeEval(right,alls,precision,recall,right_cross,predRet_list,targetRet_list):
  f1_macro = 0;
  f1_micro = 0;f1_micro_p=0;f1_micro_r=0;
  temp =  np.average(precision)+np.average(recall)
  if temp != 0:
    f1_macro = 2*np.average(precision)*np.average(recall)/(temp)*100
  
  if np.sum(predRet_list)!=0:
    f1_micro_p = np.sum(right_cross)*1.0/np.sum(predRet_list)
  if np.sum(targetRet_list)!=0:
    f1_micro_r = np.sum(right_cross)*1.0/np.sum(targetRet_list)
    
  if f1_micro_p+f1_micro_r!=0:
    f1_micro = 2*f1_micro_p*f1_micro_r/(f1_micro_p+f1_micro_r) * 100
  return 1.0 *right/alls * 100,f1_macro,f1_micro