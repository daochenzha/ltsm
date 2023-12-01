TRAIN="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf 
/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf 
/home/jy101/ltsm/dataset/monash/cif_2016_dataset.tsf 
/home/jy101/ltsm/dataset/monash/m1_monthly_dataset.tsf 
/home/jy101/ltsm/dataset/monash/m3_monthly_dataset.tsf
/home/jy101/ltsm/dataset/monash/tourism_monthly_dataset.tsf
/home/jy101/ltsm/dataset/monash/london_smart_meters_dataset_without_missing_values.tsf 
/home/jy101/ltsm/dataset/monash/kdd_cup_2018_dataset_without_missing_values.tsf 
/home/jy101/ltsm/dataset/monash/pedestrian_counts_dataset.tsf 
/home/jy101/ltsm/dataset/monash/wind_farms_minutely_dataset_without_missing_values.tsf 
/home/jy101/ltsm/dataset/monash/vehicle_trips_dataset_without_missing_values.tsf"

INIT_TEST="/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf 
/home/jy101/ltsm/dataset/monash/weather_dataset.tsf"

TEST="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf 
/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf"

PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_monash_ds"

# Train
CUDA_VISIBLE_DEVICES=0,1,2,4 python3 main_hf.py --model_id test_run --train_epochs 2 --batch_size 1000 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora_dim 256 --freeze 1

# Eval
# CUDA_VISIBLE_DEVICES=0,4,5,6 python3 main_hf.py --model_id test_run --train_epochs 1 --batch_size 1000 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora
