#@time: 2017/6/13

dir_path="data/kbp/2010"
data_tag="kbp"
dataset="training"
dims="300"


if [ ! -d "data/kbp/2010/training/features" ]; then
 mkdir "data/kbp/2010/training/features"
fi


if [ ! -d "data/kbp/2010/training/features/90" ]; then
 mkdir "data/kbp/2010/training/features/90"
fi


:<<!
generate coreference results and entMentsTags.p
!
#python utils/getCoref.py --dir_path ${dir_path}/ --data_tag ${data_tag} --dataset ${dataset}


:<<!
generate named entity recognition datasets.
!
#python embeddings/get_ace_embeddings.py --dir_path ${dir_path}/${dataset}/ --data_tag ${data_tag} --train ${dir_path}/${dataset}/${data_tag}Data.txt --sentence_length 124 --use_model data/wordvec_model_${dims}.p --model_dim ${dims}

:<<!
Named entity recognition using pre-trained NER model on CONLL datasets, very import module of our system
only has the 98 percent correctness
!
##python entityRecog.py --dir_path ${dir_path}/${dataset}/ --data_tag ${data_tag}

:<<!
Extract entity mention from NER results; we abandon the NIL entities!
#generate data/ace/features/ent_mention_index.p  [line,[[start,end,words],...]]
!
#python getNERentMentions.py --dir_path ${dir_path}/${dataset}/ --data_tag ${data_tag}

:<<!
generate candidate entities for entity mentions
!
python getACECandiates.py --dir_path ${dir_path}/${dataset}/ --data_tag ${data_tag}
##python transfer.py data/ace/ eng.ace ace.p #abandon
##python utils/getLinkingTag.py --dir_path ${dir_path} --data_tag ${data_tag} #abandon
:<<!
generate entity linking features
!
#we also need to delete the non entity mention sentences!
#python embeddings/generate_ace_entmention_linking_features.py --dir_path ${dir_path}/${dataset}/ --data_tag ${data_tag}
#python trainAidaNEL1.py

#python entityLinking.py --dir_path ${dir_path} --data_tag ${data_tag}
#python getNEL.py --dir_path ${dir_path}/${dataset}/ --data_tag ${data_tag}
