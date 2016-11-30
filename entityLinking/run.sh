#@time: 2016/11/28
#funtion: deal with CoNLL 2003 datasets

data="testa"
dir_path="data/aida"

:<<!
extract ent mention and its types
!
#python transfer.py data/aida/ eng.${data} ${data}.p

:<<!
generate entity mention candidates and related entities
!
#python getCandiates.py data/aida/process/ ${data}.p ${data}_candEnts.p


:<<!
extract raw sentence to train word2vec
!
#python getSentenceEntMent.py  data/aida/process/ train.p train_sent.txt

#python embeddings/wordvec_model.py --dir_path ${dir_path} --corpus ${dir_path}/process/train_sent.txt --dimension 100 --vocab_size 12000


:<<!
resize your data into a max sequence length: such as 50(to speed up the trianing!)
!

#python resize_input.py --input data/aida/eng.train --output ${dir_path}/process/train.out --trim 50
#python resize_input.py --input data/aida/eng.testa --output ${dir_path}/process/testa.out --trim 50
#python resize_input.py --input data/aida/eng.testb --output ${dir_path}/process/testb.out --trim 50

:<<!
get data embddings for bi-LSTM layers
!
#python embeddings/get_conll_embeddings.py --dir_path ${dir_path} --train ${dir_path}/process/train.out --test_a ${dir_path}/process/testa.out --test_b ${dir_path}/process/testb.out --use_model ${dir_path}/wordvec_model_100.p --model_dim 100
