# bert  sasrec  nextitnet nfm deepfm  GRU4Rec
data=Diginetica
model=GRU4Rec
loss=desorec
ld1=0.3
ld2=0.7

python main.py \
--class_num 16 \
--tau 2.0 \
--data_path $data.csv \
--load_path save/$data/$model/base/ \
--save_path save/$data/$model/$loss/ld1-$ld1-ld2-$ld2-c-$c-t-$t/ \
--clean 1 \
--model $model \
--cold 0 \
--samples_ratio 0.3 \
--enable_sample 0 \
--train_batch_size 256 \
--test_batch_size 256 \
--d_model 128 \
--num_epoch 20 \
--loss $loss \
--device cuda:0 \
--max_len 20 \
--lr 1e-3 \
--ld1 $ld1 \
--ld2 $ld2  


