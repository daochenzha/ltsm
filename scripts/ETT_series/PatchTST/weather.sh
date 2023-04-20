export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=PatchTST

for percent in 5 10
do
for pred_len in 96 192 336 720
do

mkdir logs/$model/$percent'_'percent
python main.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id weather_$model'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --d_model 128 \
    --n_heads 16 \
    --d_ff 512 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --itr 3 \
    --model $model > logs/$model/$percent'_'percent/weather_$model'_'$seq_len'_'$pred_len.log

done
done
done
