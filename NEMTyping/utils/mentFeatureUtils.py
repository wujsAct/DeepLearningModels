import numpy as np
'''
@sentence_final: shape:(batch_size,sequence_length,dims)
'''
def padZeros(sentence_final,max_sentence_length=80,dims=111):
  for i in range(len(sentence_final)):
    offset = max_sentence_length-len(sentence_final[i])
    sentence_final[i] += [[0]*dims]*offset
    
  return np.asarray(sentence_final)

def genEntCtxMask(args,batch_size,entment_mask_final):
  entNums = len(entment_mask_final)
  entCtxLeft_masks=[]
  entCtxRight_masks=[]
  
  for i in range(entNums):
    items = entment_mask_final[i]
    ids = items[0];start=items[1];end=items[2]
    temp_entCtxLeft_mask=[];temp_entCtxRight_mask = []
    left = max(0,start-10); 
    right = min(args.sentence_length,end+10)
    for ient in range(left,start):
        temp_entCtxLeft_mask.append(ids*args.sentence_length+ient)
    for ient in range(end,right):
        temp_entCtxRight_mask.append(ids*args.sentence_length+ient)
        
    if start-left < 10:
      temp_entCtxLeft_mask+= [batch_size*args.sentence_length] * (10-(start-left))
    
    if right-end < 10:
      temp_entCtxRight_mask+= [batch_size*args.sentence_length] * (10-(right-end))
      
    entCtxLeft_masks.append(temp_entCtxLeft_mask)
    entCtxRight_masks.append(temp_entCtxRight_mask)
  return entCtxLeft_masks,entCtxRight_masks
    
def genEntMentMask(args,batch_size,entment_mask_final):
  entNums = len(entment_mask_final)
  entment_masks = []
  #need to limit the length of the entity mentions
  for i in range(entNums):
    items = entment_mask_final[i]
    ids = items[0];start=items[1];end=items[2]
    temp_entment_masks=[]
    for ient in range(start,end):
        temp_entment_masks.append(ids*args.sentence_length+ient)
    if end-start <5:
      temp_entment_masks+= [batch_size*args.sentence_length] * (5-(end-start))
    if end-start > 5:
      temp_entment_masks = temp_entment_masks[0:5]
    entment_masks.append(list(temp_entment_masks))
  return np.asarray(entment_masks)