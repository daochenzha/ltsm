export CUDA_VISIBLE_DEVICES=0


seq_len=336
model=PatchTST

for seq_len in 336
do
for percent in 5 10
do
for pred_len in 96 192 336
do

mkdir logs/$model/$percent'_'percent
python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$model'_'336'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.000075 \
    --train_epochs 10 \
    --d_model 16 \
    --n_heads 4 \
    --d_ff 128 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --itr 3 \
    --model $model > logs/$model/$percent'_'percent/ETTh2_$model'_'$seq_len'_'$pred_len.log

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0002 \
    --train_epochs 10 \
    --d_model 16 \
    --n_heads 4 \
    --d_ff 128 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --itr 3 \
    --model $model > logs/$model/$percent'_'percent/ETTh1_$model'_'$seq_len'_'$pred_len.log

done
done
done