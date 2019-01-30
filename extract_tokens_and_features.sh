BERT_LARGE_DIR=$HOME/gits/bert/uncased_L-24_H-1024_A-16
DATA_DIR=$HOME/bert-for-chatbot/data/splitted/validation

for i in {10..10}
do
    python extract_features.py \
    --input_file=$DATA_DIR/validation_inputs_$i \
    --output_file=$DATA_DIR/validation_features_$i.json \
    --vocab_file=$BERT_LARGE_DIR/vocab.txt \
    --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
    --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
    --layers=-1,-2,-3,-4 \
    --max_seq_length=256 \
    --batch_size=16
done
