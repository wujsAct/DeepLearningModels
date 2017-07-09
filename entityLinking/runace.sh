 #@time: 2017/1/3
#function: process ace dataset
#attention: we utilize AIDA training set to train the entity mention recognition model
#step1. we uitlize this model to extract mention in ace, we evaluate the accuracy; we assume the same entity mentions in one doc refer to the same entity in KG.
#step2. we do entity linking for the extracted entity mentions
#step3. an entity mentions linking is right, when and only when entity mention recognition is right and linked entity is right!
data_tag="ace"
dir_path="data/ace/"
dims="300"


if [ ! -d "data/ace/features" ]; then
 mkdir "data/ace/features"
fi


if [ ! -d "data/ace/features/90" ]; then
 mkdir "data/ace/features/90"
fi

:<<!
generate coreference results and entMentsTags.p
!
#python utils/getCoref.py --dir_path ${dir_path} --data_tag ${data_tag} --dataset ""

:<<!
generate named entity recognition datasets.
!
#python embeddings/get_ace_embeddings.py --dir_path ${dir_path} --data_tag ${data_tag} --train ${dir_path}/aceData.txt --sentence_length 124 --use_model data/GoogleNews-vectors-negative300.bin --model_dim ${dims};

:<<!
Named entity recognition using pre-trained NER model on CONLL datasets, very import module of our system
only has the 98 percent correctness
!
#python entityRecog.py --dir_path ${dir_path}/${dataset}/ --data_tag ${data_tag}

:<<!
Extract entity mention from NER results;
generate data/ace/features/ent_mention_index.p  [line,[[start,end,words],...]]
!
#python getNERentMentions.py --dir_path ${dir_path} --data_tag ${data_tag}

:<<!
generate candidate entities for entity mentions
!
#python getACECandiates.py --dir_path ${dir_path} --data_tag ${data_tag}
##python transfer.py data/ace/ eng.ace ace.p #abandon
##python utils/getLinkingTag.py --dir_path ${dir_path} --data_tag ${data_tag} #abandon
:<<!
generate entity linking features
!
#we also need to delete the non entity mention sentences!
#python embeddings/generate_ace_entmention_linking_features.py --dir_path ${dir_path} --data_tag ${data_tag}
#python trainAidaNEL1.py

#python entityLinking.py --dir_path ${dir_path} --data_tag ${data_tag}
python getNEL.py --dir_path ${dir_path} --data_tag ${data_tag} --features 3



