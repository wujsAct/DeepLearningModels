import cPickle as pkl

class inputUtils(object):
  def __init__(self,dims,flag='train'):
    dir_path ='/home/wjs/demo/entityType/informationExtract/data/aida/features'
    if flag=='train':
      self.emb = pkl.load(open(dir_path+'/train_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/train_tag.p'+dims,'rb'))
    elif flag=='testa':
      self.emb = pkl.load(open(dir_path+'/test_a_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/test_a_tag.p'+dims,'rb'))
    else:
      self.emb = pkl.load(open(dir_path+'/test_b_embed.p'+dims,'rb'))
      self.tag = pkl.load(open(dir_path+'/test_b_tag.p'+dims,'rb'))