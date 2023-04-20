export CUDA_VISIBLE_DEVICES=0

seq_len=104
model=PatchTST

for pred_len in 24 36 48 60
do
for percent in 10
do
for lr in 0.0002
do

python main.py \
    --root_path ./datasets/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 16 \
    --n_heads 4 \
    --d_ff 128 \
    --freq 0 \
    --patch_size 24 \
    --stride 2 \
    --all 1 \
    --percent $percent \
    --itr 3 \
    --model $model > logs/$model/$percent'_'percent/illness_$model'_'$seq_len'_'$pred_len.log

done
done
done
done
