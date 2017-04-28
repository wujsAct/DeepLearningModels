Bidirectional LSTM-CRF model implemented by tensorflow    
-------------------
- Chunk evaluation:    
    1.1: CONLL 2000 datasets  data/conll2000
    1.2: feature: word embedding + POS + capital  
        + python embedding/get_conll_embedding.py --dir_path "data/conll2000/" --data_tag "testa" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 100   
    1.3: train model:
        + python trainAidaChunk_CRF.py
    
-------------------------------------------------- 
- Named entity mention boundary detection evaluation:  
    This task is very similar to Chunk task.    
    2.1 feature  word embedding + POS + capital     
      + python embedding/get_ner_embeddings.py --dir_path "data/conll2003/" --data_tag "testa" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 100   
      + python embedding/get_ner_embeddings.py --dir_path "data/figer_test/" --data_tag "figer" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 100   
    2.2 train: CONLL 2003 datasets data/conll2003/train;     
        we transfer the tag into B-E,I-E,O to train the NEM boundary detection   
        + python trainAidaNER_CRF.py    
    2.3 test: figer_test data/figer_test, data/conll2003/eng.testa dta/conll2003/eng.testb    
----------------------
- Named entity typing:   
    train: CONLL2003: Persion, Location, Organization, MISC  
    3.1 feature   
        word embedding + POS + capital + chunk     
    3.2 detection and typing combination    
        tag is 'I-PER', 'I-LOC', 'I-ORG', 'MISC', 'O'
        python embedding/get_conll_embedding.py --dir_path "data/conll2003/" --data_tag "testa" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 100 
        + python trainAidaNER.py 
    3.3 first detection and then do entity typing   
        
----------------------------------
- Fine-grained named entity typing   
    4.1 feature   
    word embedding + POS + capital + chunk   
    4.2 train: data/figer      
      + python trainAidaNERFiger.py    #get_input_figer_chunk_train(in file embedding/get_ner_embedding.py) to get the figer train dataset features   
    4.3 test: data/figer_test   
    