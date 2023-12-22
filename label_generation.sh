# bert  sasrec  nextitnet nfm deepfm  GRU4Rec
data=Diginetica
model=GRU4Rec


python LE_train.py \
--dataname $data \
--class_num 16 \
--tau 2.0 \
--data_path $data.csv \
--load_path save/$data/$model/base/ \
--save_path save/$data/$model/base/ \
--model $model \
--train_batch_size 256 \
--test_batch_size 256 \
--d_model 128 \
--device cuda:0 \
--max_len 20 \
--lr 1e-3 \
--get_user 1


