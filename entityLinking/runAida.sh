#@time: 2016/11/28
#funtion: deal with CoNLL 2003 datasets, 数据中存在很多问题呢！
data_train="train"
data_testb="testb"
data_testa="testa"
dir_path="data/aida/"
dims="100"

:<<!
Step1: extract ent mention and its types
!
#python transfer.py data/aida/ eng.train train.p train.out train_sentid2aNosNoid.txt train_entaNosNoid.txt
#python transfer.py data/aida/ eng.testa testa.p testa.out testa_sentid2aNosNoid.txt testa_entaNosNoid.txt
#python transfer.py data/aida/ eng.testb testb.p testb.out testb_sentid2aNosNoid.txt testb_entaNosNoid.txt

:<<!
Step2: get data embddings for bi-LSTM layers
!
#rm -rf ${dir_path}/features;
#mkdir ${dir_path}/features;
#time:2017/1/9 revise train data into TFRecord, so that we can do 
#
python embeddings/get_conll_embeddings.py --dir_path ${dir_path} --data_train ${dir_path}/process/train.p --data_testa ${dir_path}/process/testa.p --data_testb ${dir_path}/process/testb.p --train ${dir_path}/features/train.out --test_a ${dir_path}/features/testa.out --test_b ${dir_path}/features/testb.out --use_model data/wordvec_model_${dims}.p --model_dim ${dims}  --sentence_length 250

#python trainAidaNER.py
#python entityRecog.py --dir_path ${dir_path} --data_tag testb
#python entityRecog.py --dir_path ${dir_path} --data_tag testa

:<<!
generate entity mention candidates and related entities
!
#python getCandiates.py data/aida/process/ testb.p testb_candEnts.p
#python getCandiates.py data/aida/process/ train.p train_candEnts.p
#python getCandiates.py data/aida/process/ testa.p testa_candEnts.p

:<<!
step3: generate entity linking tag data/aida/aida-annotation.p, and reivse train.p, train_entms.p100 to train.p_new, train_entms.p100_new
!
#python python utils/readAIDAAnnotation.py

:<<!
genenrate wid2fbId and wtitle2fbId 
!
#python utils/wiki2fb.py

:<<!
generate freebase mid2type, generate data/mid2types.txt
!
#python utils/readfb.py
:<<!
extract raw sentence to train word2vec
!
#python getSentenceEntMent.py  data/aida/process/ train.p  train_candEnts.p train_sent.txt train_candEnt_descrip_dict.p train
#python getSentenceEntMent.py  data/aida/process/ testa.p testa_candEnts.p testa_sent.txt testa_candEnt_descrip_dict.p testa
#python getSentenceEntMent.py  data/aida/process/ testb.p testb_candEnts.p testb_sent.txt testb_candEnt_descrip_dict.p testb
#cd data/aida/process/; cat testa_sent.txt testb_sent.txt train_sent.txt > total_sent.txt;cd ..;cd ..;cd ..;
#python embeddings/wordvec_model.py --dir_path ${dir_path} --corpus ${dir_path}/process/total_sent.txt --dimension ${dims} #--vocab_size 23661

#python trainAidaNEL1.py
:<<!
generate entity linking tag data/aida/aida-annotation.p
!
#cd utils; python readAIDAAnnotation.py; cd ..

:<<!
step4: co-reference resolution ,generate process/testa_entMent2repMent.p {aNosNo\tstartIndex\tendIndex\tmention:representive entity aNosNo\tstartIndex\tendIndex\tmention}
!
#python utils/getAidaCoref.py --dir_path ${dir_path} --data_tag testa
#python utils/getAidaCoref.py --dir_path ${dir_path} --data_tag testb
#python utils/getAidaCoref.py --dir_path ${dir_path} --data_tag train


:<<!
generate all candiate entities to entity mentions. flag testa: annotation train = 0, ent_id = 23396, testb = 29313 
!
#python getctxCnnData.py ${dir_path} testa_candEnts.p testa_entms.p100 testa_ent_cand_mid.p testa
#python getctxCnnData.py ${dir_path} testb_candEnts.p testb_entms.p100 testb_ent_cand_mid.p testb
#python getctxCnnData.py ${dir_path} train_candEnts.p train_entms.p100_new train_ent_cand_mid.p train

:<<!
generate testa_ent_cand_mid_new.p: run getNELAida function: printfeatues
!
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_testa}
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_testb}
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_train}

:<<!
generate entity linking features cPickle
!
#python embeddings/generate_entmention_linking_features.py ${dir_path} testa_entms.p100 testa_embed.p100 testa_ent_cand_mid_new.p testa_ent_linking.p testa_ent_linking_type.p testa_ent_linking_candprob.p testa_ent_relcoherent.p testa_ent_mentwordv.p testa
#python embeddings/generate_entmention_linking_features.py ${dir_path} testb_entms.p100 test_b_embed.p100 testb_ent_cand_mid_new.p testb_ent_linking.p testb_ent_linking_type.p testb_ent_linking_candprob.p testb_ent_relcoherent.p testb_ent_mentwordv.p testb
#python embeddings/generate_entmention_linking_features.py ${dir_path} train_entms.p100_new train_embed.p100 train_ent_cand_mid_new.p train_ent_linking.p train_ent_linking_type.p train_ent_linking_candprob.p train_ent_relcoherent.p train_ent_mentwordv.p train

:<<!
#time:2017/1/10 revise entity linking train data into TFRecord! train.tfrecord
!
#python generateNELfeature.py 

#python trainAidaNEL1.py
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_testa}
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_testb}
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_train}

:<<!
generate similar entity mention!
!
#python entityLinking.py --dir_path ${dir_path} --data_tag ${data_testa} #get testa entity mention encoder!
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_testa}  #run function:
#python getNELAida.py --dir_path ${dir_path} --data_tag ${data_testb}

#python trainAidaNEL2.py --features 0
#python trainAidaNEL2.py --features 1_1
#python trainAidaNEL2.py --features 1_2
#python trainAidaNEL2.py --features 2
#python trainAidaNEL2.py --features 3