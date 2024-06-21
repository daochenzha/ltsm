TRAIN="./datasets/ETT-small/ETTh1.csv 
    ./datasets/ETT-small/ETTh2.csv
    ./datasets/ETT-small/ETTm1.csv
    ./datasets/ETT-small/ETTm2.csv
    ./datasets/electricity/electricity.csv
    ./datasets/exchange_rate/exchange_rate.csv
    ./datasets/traffic/traffic.csv
    ./datasets/weather/weather.csv"

INIT_TEST="/home/sl237/all_six_datasets/electricity/electricity.csv 
/home/sl237/all_six_datasets/weather/weather.csv"

TRAIN="./datasets/ETT-small/ETTh1.csv 
    ./datasets/ETT-small/ETTh2.csv
    ./datasets/ETT-small/ETTm1.csv
    ./datasets/ETT-small/ETTm2.csv
    ./datasets/electricity/electricity.csv
    ./datasets/exchange_rate/exchange_rate.csv
    ./datasets/traffic/traffic.csv
    ./datasets/weather/weather.csv"

PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_csv_split"


for pred_len in 96 192 336 720
do
for lr in 1e-3 
    for lora_dim in 32 64
    do
        do
            epoch=500
            downsample_rate=20
            freeze=0
            OUTPUT_PATH="output/ltsmt_new_csv_large_lr${lr}_loraFalse_down${downsample_rate}_freeze${freeze}_e${epoch}_pred${pred_len}/"
            CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 main_hf.py \
                --lora \
                --lora_dim ${lora_dim} \
                --model_id test_run \
                --train_epochs ${epoch} \
                --batch_size 800 \
                --pred_len ${pred_len} \
                --gradient_accumulation_steps 64 \
                --data_path ${TRAIN} \
                --test_data_path ${INIT_TEST} \
                --test_data_path_list ${TEST} \
                --prompt_data_path ${PROMPT} \
                --freeze ${freeze} \
                --learning_rate ${lr} \
                --downsample_rate ${downsample_rate} \
                --output_dir ${OUTPUT_PATH}
        done
    done
done