#@time: 2016/11/28
#funtion: deal with CoNLL 2003 datasets

data="testa"
dir_path="data/aida"
dims="100"
:<<!
extract ent mention and its types
!
#python transfer.py data/aida/ eng.train train.p
#python transfer.py data/aida/ eng.testa testa.p
#python transfer.py data/aida/ eng.testb testb.p

:<<!
generate entity mention candidates and related entities
!
#python getCandiates.py data/aida/process/ train.p train_candEnts.p
#python getCandiates.py data/aida/process/ testa.p testa_candEnts.p
#python getCandiates.py data/aida/process/ testb.p testb_candEnts.p


:<<!
extract raw sentence to train word2vec
!
#python getSentenceEntMent.py  data/aida/process/ train.p train_sent.txt
#python getSentenceEntMent.py  data/aida/process/ testa.p testa_sent.txt
#python getSentenceEntMent.py  data/aida/process/ testb.p testb_sent.txt
#cd data/aida/process/; cat testa_sent.txt testb_sent.txt train_sent.txt > total_sent.txt;cd ..;cd ..;cd ..;
#python embeddings/wordvec_model.py --dir_path ${dir_path} --corpus ${dir_path}/process/total_sent.txt --dimension ${dims} #--vocab_size 23661


:<<!
resize your data into a max sequence length: such as 50(to speed up the trianing!)
!
#python resize_input.py --input data/aida/eng.train --entMentInput data/aida/process/train.p --output ${dir_path}/process/train.out --trim 50
#python resize_input.py --input data/aida/eng.testa --entMentInput data/aida/process/testa.p --output ${dir_path}/process/testa.out --trim 50
#python resize_input.py --input data/aida/eng.testb --entMentInput data/aida/process/testb.p --output ${dir_path}/process/testb.out --trim 50

:<<!
get data embddings for bi-LSTM layers
!
python embeddings/get_conll_embeddings.py --dir_path ${dir_path} --data_train ${dir_path}/process/train.p --data_testa ${dir_path}/process/testa.p --data_testb ${dir_path}/process/testb.p --train ${dir_path}/process/train.out --test_a ${dir_path}/process/testa.out --test_b ${dir_path}/process/testb.out --use_model ${dir_path}/wordvec_model_${dims}.p --model_dim ${dims}


:<<!
generate cnn embedding datasets.
!
#python getctxCnnData.py ${dir_path} testa_candEnts.p 