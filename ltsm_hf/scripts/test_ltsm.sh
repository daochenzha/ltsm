TRAIN="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf"

TEST="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf 
/home/jy101/ltsm/dataset/monash/australian_electricity_demand_dataset.tsf"

INIT_TEST="/home/jy101/ltsm/dataset/monash/weather_dataset.tsf"

PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_monash_ds"


PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_monash_ds"

# OUTPUT_PATH="output/ltsm_train_lr0005_down270/"
# Train
pretrain="output/ltsm_train_large_lr1e-3_loraFalse_down180_freeze0_e2000"
pretrained="/home/sl237/ltsm/ltsm_hf/output/ltsm_train_large_lr1e-3_loraFalse_down180_freeze0_e2000/checkpoint-525"

epoch=2000
downsample_rate=180
freeze=1
eval=1
OUTPUT_PATH="output/test_monash_ltsm_train_large_lr1e-3_loraFalse_down180_freeze0_e2000/checkpoint-525"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 main_hf.py --model_id test_run --train_epochs ${epoch} --batch_size 800 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --freeze ${freeze} --downsample_rate ${downsample_rate} --test_data_path ${INIT_TEST}  --output_dir ${OUTPUT_PATH}  --downsample_rate ${downsample_rate} --output_dir ${OUTPUT_PATH} --eval ${eval} --local_pretrain ${pretrained}