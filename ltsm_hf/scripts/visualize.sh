TRAIN="/home/sl237/all_six_datasets/ETT-small/ETTh1.csv"

INIT_TEST="/home/sl237/all_six_datasets/electricity/electricity.csv"

TEST="/home/sl237/all_six_datasets/electricity/electricity.csv"

PROMPT="/home/gw22/python_project/ltsm_proj/ltsm/prompt/prompt_data_normalize_csv"
pretrained="/home/sl237/ltsm/ltsm_hf/output/ltsm_csv_large_lr1e-3_loraFalse_down10_freeze0_e1000/checkpoint-1425"
# OUTPUT_PATH="output/ltsm_train_lr0005_down270/"
# Train
epoch=2000
downsample_rate=180
freeze=1
eval=1
OUTPUT_PATH="output/visualize_ltsm_csv_large_lr1e-3_loraFalse_down10_freeze0_e1000/checkpoint-1425"
CUDA_VISIBLE_DEVICES=0 python3 main_hf.py --model_id test_run --train_epochs ${epoch} --batch_size 800 --gradient_accumulation_steps 64 --data_path ${TRAIN} --test_data_path ${INIT_TEST} --test_data_path_list ${TEST} --prompt_data_path ${PROMPT} --freeze ${freeze} --downsample_rate ${downsample_rate} --test_data_path ${INIT_TEST}  --output_dir ${OUTPUT_PATH}  --downsample_rate ${downsample_rate} --output_dir ${OUTPUT_PATH} --eval ${eval} --local_pretrain ${pretrained}

