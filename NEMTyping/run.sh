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
#python embedding/get_ner_embeddings.py --dir_path "data/conll2003/" --data_tag "testa" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 300
#python embedding/get_ner_embeddings.py --dir_path "data/conll2003/" --data_tag "testb" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 300
#python embedding/get_ner_embeddings.py --dir_path "data/conll2003/" --data_tag "train" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 300
##python embedding/get_ner_embeddings.py --dir_path "data/figer/" --data_tag "figer" --use_model data/wordvec_model_${dims}.p --sentence_length 80 --model_dim 100
#python embedding/get_ner_embeddings.py --dir_path "data/figer_test/" --data_tag "figer" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 300

#python embedding/get_ner_embeddings.py --dir_path "data/WebQuestion/" --data_tag "test" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 300

#python embedding/get_ner_embeddings.py --dir_path "data/WebQuestion/" --data_tag "train" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 300

#python embedding/get_ner_embeddings.py --dir_path "data/ace/" --data_tag "ace" --use_model data/wordvec_model_${dims}.p --sentence_length 124 --model_dim 300

#python trainAidaNER_CRF.py

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


:<<!
we utilize utils/splitFiger.py to split the Figer train into train_set, validation_set and test_set
we generate the data/figer/features/figerData_testa.txt and testa_entMents.p
!
#python utils/splitFiger.py
#
#parser.add_argument('--dataset', type=str, help='[figer, OntoNotes]', required=True)
#parser.add_argument('--wordEmbed', type=str, help='[LSTM,MLP]', required=True)
#parser.add_argument('--batch_size', type=int, help='[1000 for figer; 1500 for OntoNotes]', required=True)
#parser.add_argument('--class_size', type=int, help='[113 for figer; 89 for OntoNotes]', required=True)
#parser.add_argument('--model', type=str, help='[0:seqCNN,1:seqMLP,2:seqCtxLSTM,3:seqLSTM]', required=True)
#parser.add_argument('--sentence_length', type=int, help='[250 for OntoNotes; 62 for figer]', required=True)
#parser.add_argument('--iterateEpoch', type=int, help='[10 for figer, 1 for OntoNotes]', required=True)

python trainAidaNERFiger.py --dataset figer --wordEmbed 300 --batch_size 1000 --class_size 113 --model 0 --sentence_length 62 --iterateEpoch 5
#python trainAidaNERFiger.py --dataset figer --wordEmbed 300 --batch_size 1000 --class_size 113 --model 1 --sentence_length 62 --iterateEpoch 1
#python trainAidaNERFiger.py --dataset figer --wordEmbed 300 --batch_size 1000 --class_size 113 --model 3 --sentence_length 62 --iterateEpoch 1
