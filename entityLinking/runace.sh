#@time: 2017/1/3
#function: process ace dataset
#attention: we utilize AIDA training set to train the entity mention recognition model
#step1. we uitlize this model to extract mention in ace, we evaluate the accuracy; we assume the same entity mentions in one doc refer to the same entity in KG.
#step2. we do entity linking for the extracted entity mentions
#step3. an entity mentions linking is right, when and only when entity mention recognition is right and linked entity is right!
data="ace"
dir_path="data/ace"
dims="100"

:<<!
generate named entity recognition datasets.
!
#rm -rf data/ace/features;
#mkdir data/ace/features;
python embeddings/get_ace_embeddings.py --dir_path ${dir_path} --train ${dir_path}/aceData.txt --use_model data/wordvec_model_${dims}.p --model_dim ${dims};
