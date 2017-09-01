#@time: 2017/8/4
#we utilize the new model to solve entity linking

data="testa"
dir_path="data/aida/"
dims="100"

:<<!
extract all candidate entity
!
python newSteps/extractAllCandidatesEnts.py

:<<!
we utilize co-occurence to train the embeddings for all candidate entities
!


:<<!
generate entity mention candidates and related entities
!
#python getCandiates.py data/aida/process/ testb.p testb_candEnts.p
#python getCandiates.py data/aida/process/ train.p train_candEnts.p
#python getCandiates.py data/aida/process/ testa.p testa_candEnts.p
#python getLinkTags.py data/aida/process/ train.p train_candEnts.p
#python getLinkTags.py data/aida/process/ testa.p testa_candEnts.p
#python getLinkTags.py data/aida/process/ testb.p testb_candEnts.p

:<<!
genenrate wid2fbId and wtitle2fbId
!
python utils/wiki2fb.py
python utils/reverseIndexUtils.py

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


:<<!
resize your data into a max sequence length: such as 124(to speed up the trianing!) max sequence length of testa
!
#python resize_input.py --input data/aida/eng.train --entMentInput data/aida/process/train.p --output ${dir_path}/process/train.out --trim 124
#python resize_input.py --input data/aida/eng.testa --entMentInput data/aida/process/testa.p --output ${dir_path}/process/testa.out --trim 124
#python resize_input.py --input data/aida/eng.testb --entMentInput data/aida/process/testb.p --output ${dir_path}/process/testb.out --trim 124

:<<!
get data embddings for bi-LSTM layers
!
#rm -rf ${dir_path}/features;
#mkdir ${dir_path}/features;
#time:2017/1/9 revise train data into TFRecord, so that we can do 
#
#python embeddings/get_conll_embeddings.py --dir_path ${dir_path} --data_train ${dir_path}/process/train.p --data_testa ${dir_path}/process/testa.p --data_testb ${dir_path}/process/testb.p --train ${dir_path}/process/train.out --test_a ${dir_path}/process/testa.out --test_b ${dir_path}/process/testb.out --use_model data/wordvec_model_${dims}.p --model_dim ${dims}  --sentence_length 124

#python embeddings/getNERSequenceLength.py --dir_path ${dir_path} --model_dim ${dims}

#python trainAidaNER.py

#python trainAidaNEL1.py
:<<!
generate entity linking tag data/aida/aida-annotation.p
!
#cd utils; python readAIDAAnnotation.py; cd ..

:<<!
generate all candiate entities to entity mentions. flag testa: annotation train = 0, ent_id = 23396, testb = 29313 
!
#python getctxCnnData.py ${dir_path} testa_candEnts.pnew testa_entms.p100 testa_ent_cand_mid.p testa
#python getctxCnnData.py ${dir_path} testb_candEnts.pnew testb_entms.p100 testb_ent_cand_mid.p testb
#python getctxCnnData.py ${dir_path} train_candEnts.pnew train_entms.p100 train_ent_cand_mid.p train

:<<!
generate entity linking features cPickle
!
#python embeddings/generate_entmention_linking_features.py ${dir_path} testa_entms.p100 test_a_embed.p100 testa_ent_cand_mid.p testa_ent_linking.p testa_ent_linking_type.p testa_ent_linking_candprob.p testa_ent_relcoherent.p testa_ent_mentwordv.p testa
#python embeddings/generate_entmention_linking_features.py ${dir_path} testb_entms.p100 test_b_embed.p100 testb_ent_cand_mid.p testb_ent_linking.p testb_ent_linking_type.p testb_ent_linking_candprob.p testb_ent_relcoherent.p testb_ent_mentwordv.p testb
#python embeddings/generate_entmention_linking_features.py ${dir_path} train_entms.p100 train_embed.p100 train_ent_cand_mid.p train_ent_linking.p train_ent_linking_type.p train_ent_linking_candprob.p train_ent_relcoherent.p train_ent_mentwordv.p train

:<<!
#time:2017/1/10 revise entity linking train data into TFRecord! train.tfrecord
!
#python generateNELfeature.py 

#python trainAidaNEL1.py
