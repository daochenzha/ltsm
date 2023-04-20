export CUDA_VISIBLE_DEVICES=0


seq_len=512
model=PatchTST

for percent in 5
do
for pred_len in 96 192 336 720
do

mkdir logs/$model/$percent'_'percent
python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$model'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_m \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 128 \
    --n_heads 16 \
    --d_ff 512 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --itr 3 \
    --model $model > logs/$model/$percent'_'percent/ETTm1_$model'_'$seq_len'_'$pred_len.log

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_$model'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_m \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 128 \
    --n_heads 16 \
    --d_ff 512 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --itr 3 \
    --model $model > logs/$model/$percent'_'percent/ETTm2_$model'_'$seq_len'_'$pred_len.log

done
done
done
