TRAIN="/home/yc146/ltsm/dataset/monash/weather_dataset.tsf /home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf"
INIT_TEST="/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf"
TEST="/home/yc146/ltsm/dataset/monash/weather_dataset.tsf /home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf"
PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize"

# Train
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_hf.py --model_id test_run --train_epochs 600 --batch_size 1000 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora_dim 256 --freeze 0

# Eval
# CUDA_VISIBLE_DEVICES=0,4,5,6 python3 main_hf.py --model_id test_run --train_epochs 1 --batch_size 1000 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --lora
