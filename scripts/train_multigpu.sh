i="eeg_train" #dataset/traffic.csv dataset/weather.csv dataset/electricity.csv
CUDA_VISIBLE_DEVICES="1,3" accelerate launch --multi_gpu main.py --model_id test_run --data custom_list --train_epochs 10 --batch_size 32 --gradient_accumulation_steps 2 --data_path ${i}  #> log.eeg

# CUDA_VISIBLE_DEVICES="1,3" accelerate launch --multi_gpu main.py --model_id test_run