TRAIN="/home/sl237/all_six_datasets/ETT-small/ETTh1.csv"

INIT_TEST="/home/sl237/all_six_datasets/electricity/electricity.csv 
/home/sl237/all_six_datasets/weather/weather.csv"

TEST="/home/sl237/all_six_datasets/ETT-small/ETTh1.csv 
    /home/sl237/all_six_datasets/ETT-small/ETTh2.csv
    /home/sl237/all_six_datasets/ETT-small/ETTm1.csv
    /home/sl237/all_six_datasets/ETT-small/ETTm2.csv
    /home/sl237/all_six_datasets/electricity/electricity.csv
    /home/sl237/all_six_datasets/exchange_rate/exchange_rate.csv
    /home/sl237/all_six_datasets/traffic/traffic.csv
    /home/sl237/all_six_datasets/weather/weather.csv"

PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_csv_split"
pretrained="/home/sl237/ltsm/ltsm_hf/output/ltsm_train_large_lr1e-3_loraFalse_down180_freeze0_e2000/checkpoint-525"


epoch=2000
downsample_rate=180
freeze=1
eval=1
OUTPUT_PATH="output/deltest_ltsm_train_large_lr1e-3_loraFalse_down180_freeze0_e2000/checkpoint-525"
CUDA_VISIBLE_DEVICES=0 python3 main_hf.py --model_id test_run --train_epochs ${epoch} --batch_size 800 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --freeze ${freeze} --downsample_rate ${downsample_rate} --test_data_path ${INIT_TEST}  --output_dir ${OUTPUT_PATH}  --downsample_rate ${downsample_rate} --output_dir ${OUTPUT_PATH} --eval ${eval} --local_pretrain ${pretrained}

