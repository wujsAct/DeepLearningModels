Bidirectional LSTM-CRF model implemented by temsorflow

1. Chunk evaluation: 

    CONLL 2000 datasets  data/conll2000

    feature: word embedding + POS + capital

    
2. Named entity mention boundary detection evaluation:

    This task is very similar to Chunk task.

    2.1 feature  word embedding + POS + capital

    2.2 train: CONLL 2003 datasets data/conll2003/train;
 
        we transfer the tag into B-E,I-E,O to train the NEM boundary detection 

    2.3 test: figer_test data/figer_test, data/conll2003/eng.testa dta/conll2003/eng.testb  


3. Named entity typing:

    train: CONLL2003: Persion, Location, Organization, MISC

    3.1 detection and typing combination 

   
    3.2 first detection and then do entity typing


4. Fine-grained named entity typing (first detection and then typing)

    train: data/figer

    test: data/figer_test