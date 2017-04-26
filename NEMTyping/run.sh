#generate chunk features
dims="100"
:<<!
chunk: conll2000
!
#python embedding/get_conll_embeddings.py --dir_path "data/conll2000/" --data_tag "train" --use_model /home/wjs/demo/entityType/informationExtract/data/wordvec_model_${dims}.p --sentence_length 80 --model_dim 100
#python embedding/get_conll_embeddings.py --dir_path "data/conll2000/" --data_tag "test" --use_model /home/wjs/demo/entityType/informationExtract/data/wordvec_model_${dims}.p --sentence_length 80 --model_dim 100

:<<!
NER: conll2003
!
#python embedding/get_ner_embeddings.py --dir_path "data/conll2003/" --data_tag "testa" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 100
#python embedding/get_ner_embeddings.py --dir_path "data/conll2003/" --data_tag "testb" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 100
#python embedding/get_ner_embeddings.py --dir_path "data/conll2003/" --data_tag "train" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 100
#python embedding/get_ner_embeddings.py --dir_path "data/figer/" --data_tag "figer" --use_model data/wordvec_model_${dims}.p --sentence_length 80 --model_dim 100
python embedding/get_ner_embeddings.py --dir_path "data/figer_test/" --data_tag "figer" --use_model data/wordvec_model_${dims}.p --sentence_length 80 --model_dim 100


#python entityRecog.py --dir_path "data/figer/" --data_tag "figer"
#python entityRecog.py --dir_path "data/figer_test/" --data_tag "figer"

:<<!
train conll2003 ner task
!
#python trainAidaNER_CRF.py

:<<!
generate the figer ner_crf results
!
#python entityRecog.py --dir_path "data/figer/" --data_tag "figer"
