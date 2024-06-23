TRAIN="datasets/ETT-small/ETTh1.csv 
    datasets/ETT-small/ETTh2.csv
    datasets/ETT-small/ETTm1.csv
    datasets/ETT-small/ETTm2.csv
    datasets/electricity/electricity.csv
    datasets/exchange_rate/exchange_rate.csv
    datasets/traffic/traffic.csv
    datasets/weather/weather.csv"

TEST="datasets/ETT-small/ETTh1.csv 
    datasets/ETT-small/ETTh2.csv
    datasets/ETT-small/ETTm1.csv
    datasets/ETT-small/ETTm2.csv
    datasets/electricity/electricity.csv
    datasets/exchange_rate/exchange_rate.csv
    datasets/traffic/traffic.csv
    datasets/weather/weather.csv"
PROMPT="prompt_bank/prompt_data_normalize_csv_split"
lr=1e-3
epoch=50
downsample_rate=20
freeze=0
d_ff=128 
OUTPUT_PATH="output/ltsm_tokenizer_lr${lr}_loraFalse_down${downsample_rate}_freeze${freeze}_e${epoch}_pred${pred_len}/"

for pred_len in 96 192 336 720
do
    CUDA_VISIBLE_DEVICES=0,1 python3 main_tokenizer.py \
    --model LTSM_Tokenizer \
    --model_name_or_path gpt2-medium \
    --d_ff $d_ff \
    --train_epochs ${epoch} \
    --batch_size 20 \
    --pred_len ${pred_len} \
    --gradient_accumulation_steps 64 \
    --data_path ${TRAIN} \
    --test_data_path_list ${TEST} \
    --prompt_data_path ${PROMPT} \
    --freeze ${freeze} \
    --learning_rate ${lr} \
    --downsample_rate ${downsample_rate} \
    --output_dir ${OUTPUT_PATH}\
    --eval 0
done
