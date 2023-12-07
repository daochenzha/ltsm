TRAIN="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf 
    /home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf 
    /home/jy101/ltsm/dataset/monash/cif_2016_dataset.tsf 
    /home/jy101/ltsm/dataset/monash/tourism_monthly_dataset.tsf
    /home/jy101/ltsm/dataset/monash/london_smart_meters_dataset_without_missing_values.tsf 
    /home/jy101/ltsm/dataset/monash/kdd_cup_2018_dataset_without_missing_values.tsf
    "

INIT_TEST="/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf 
/home/jy101/ltsm/dataset/monash/weather_dataset.tsf"

TEST="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf 
/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf"

PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_monash_ds"

OUTPUT_PATH="output/ltsm_train_lr0005_down270/"
# Train
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 main_hf.py --model_id test_run --train_epochs 400 --batch_size 1400 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora_dim 256 --freeze 1 --learning_rate 0.0005 --downsample_rate 270 --output_dir ${OUTPUT_PATH}

# Eval
# CUDA_VISIBLE_DEVICES=0,4,5,6 python3 main_hf.py --model_id test_run --train_epochs 1 --batch_size 1000 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora
