data=Diginetica
model=GRU4Rec
python main.py \
--data_path $data.csv \
--load_path save/$data/$model/base/ \
--save_path save/$data/$model/base/ \
--model $model \
--train_batch_size 256 \
--test_batch_size 256 \
--d_model 128 \
--num_epoch 20 \
--loss ce \
--device cuda:0 \
--max_len 20 \
--lr 1e-3 \

