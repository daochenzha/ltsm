export CUDA_VISIBLE_DEVICES=0

seq_len=336
model=GPT4TS

for gpt_layer in 6 0
do
for percent in 5 10
do
for pred_len in 96
do
for tmax in 20
do

mkdir logs/$model/$percent'_'percent

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --decay_fac 0.5 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 1 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax $tmax \
    --pretrain 1 \
    --is_gpt 1 > logs/$model/$percent'_'percent/ETTh2_$model'_'$gpt_layer'_'$seq_len'_'$pred_len.log


done
done
done
done


for gpt_layer in 6 0
do
for percent in 5 10
do
for pred_len in 192 336 720
do
for tmax in 20
do

mkdir logs/$model/$percent'_'percent

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --decay_fac 0.5 \
    --learning_rate 0.0000005 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 1 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 3 \
    --model $model \
    --cos 0 \
    --tmax $tmax \
    --pretrain 1 \
    --is_gpt 1 > logs/$model/$percent'_'percent/ETTh2_$model'_'$gpt_layer'_'$seq_len'_'$pred_len.log


done
done
done
done