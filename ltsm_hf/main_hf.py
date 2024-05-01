import numpy as np
import torch
from torch import nn
import os
import argparse
import random
import ipdb

from tsbench.data_pipeline.data_factory import get_data_loaders, get_datasets
from ltsm.data_provider.hf_data_loader import HF_Dataset
from ltsm.models import get_model, LTSMConfig
from peft import get_peft_config, get_peft_model, LoraConfig

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)

def get_args():
    parser = argparse.ArgumentParser(description='LTSM')

    parser.add_argument('--model_id', type=str, required=True, default='test_run')
    parser.add_argument('--model_name_or_path', type=str, default="gpt2-medium") # google/gemma-2b, meta-llama/Llama-2-7b-hf
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--data_path', nargs='+', default='dataset/weather.csv')
    # parser.add_argument('--data_path', type=str, default='dataset/weather.csv')
    parser.add_argument('--test_data_path', type=str, default='dataset/weather.csv')
    parser.add_argument('--test_data_path_list', nargs='+', required=True)
    parser.add_argument('--prompt_data_path', type=str, default='./weather.csv')
    parser.add_argument('--data_processing', type=str, default="standard_scaler")
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=str, default="h")
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)

    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--prompt_len', type=int, default=133)
    parser.add_argument('--test_seq_len', type=int, default=512)
    parser.add_argument('--test_pred_len', type=int, default=96)
    parser.add_argument('--test_label_len', type=int, default=48)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)

    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--local_pretrain', type=str, default="None")
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--eval', type=int, default=0)

    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--cos', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output/ltsm_train_lr0005/')
    parser.add_argument('--lora', action="store_true")
    parser.add_argument('--lora_dim', type=int, default=128)
    parser.add_argument('--downsample_rate', type=int, default=100)
    parser.add_argument('--llm_layers', type=int, default=32)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args


def seed_all(fixed_seed):
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)

def freeze_parameters(model):

    freeze_param_buf = ["gpt2"]
    for n, p in model.named_parameters():
        if any(fp in n for fp in freeze_param_buf):
            p.requires_grad = False
            print(f"{n} has been freeezed")

    trainable_param_buf = ["ln", "wpe", "in_layer", "out_layer", "lora"]
    for n, p in model.named_parameters():
        if any(fp in n for fp in trainable_param_buf):
            p.requires_grad = True

def print_trainable_parameters(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n} is trainable...")

    # import pdb
    # pdb.set_trace()

def run(args):
    print(args)
    model_config = LTSMConfig(**vars(args))
    model = get_model(model_config)

    if args.lora:
        peft_config = LoraConfig(
            target_modules=["c_attn"],  # ["q", "v"],
            inference_mode=False,
            r=args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif args.freeze:
        freeze_parameters(model)

    print_trainable_parameters(model)

    # TODO warmup step & lower lr
    model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(model_optim, T_0=10, T_mult=2, eta_min=1e-8)
    
    # early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        if preds.shape != p.label_ids.shape:
            label_ids = np.squeeze(p.label_ids)
        else:
            label_ids = p.label_ids
        return {
                "mse": ((preds - label_ids) ** 2).mean().item(),
                "mae": (np.abs(preds - label_ids)).mean().item()}

    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(inputs["input_data"])
        loss = nn.functional.mse_loss(outputs, inputs["labels"])
        return (loss, outputs) if return_outputs else loss

    def collate_fn(batch):
        return {
            'input_data': torch.from_numpy(np.stack([x['input_data'] for x in batch])).type(torch.float32),
            'labels': torch.from_numpy(np.stack([x['labels'] for x in batch])).type(torch.float32),
        }

    @torch.no_grad()
    def prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None):
        # CSV
        input_data = inputs["input_data"].to(model.device)
        labels = inputs["labels"].to(model.device)
        
        # monash
        # input_data = inputs["input_data"].to(model.device)
        # labels = inputs["labels"].to(model.device)
        outputs = model(input_data)
        loss = nn.functional.mse_loss(outputs, labels)
        return (loss, outputs, labels)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=args.train_epochs,
        fp16=False,
        save_steps=100,
        eval_steps=25,
        logging_steps=5,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=10,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
    )

    # Training settings
    train_dataset, eval_dataset, test_dataset, _ = get_datasets(args)
    train_dataset, eval_dataset, test_dataset = HF_Dataset(train_dataset), HF_Dataset(eval_dataset), HF_Dataset(test_dataset)

    # from transformers import AutoTokenizer, PatchTSTForPrediction
    # model = PatchTSTForPrediction.from_pretrained("namctin/patchtst_etth1_forecast")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=None,
        optimizers=(model_optim, lr_scheduler),
    )

    # Overload the trainer API
    if not args.eval:
        trainer.compute_loss = compute_loss
        trainer.prediction_step = prediction_step        
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

    # Testing settings
    for data_path in args.test_data_path_list:
        trainer.compute_loss = compute_loss
        trainer.prediction_step = prediction_step   
        args.test_data_path = data_path
        _, _, test_dataset, _ = get_datasets(args)
        test_dataset = HF_Dataset(test_dataset)
        # data_pred = trainer.predict(test_dataset)
        

        metrics = trainer.evaluate(test_dataset)
        # ipdb.set_trace()
        # np.save('/home/sl237/ltsm/ltsm_hf/scripts/' + 'preds.npy', metrics['preds'])
        # np.save('/home/sl237/ltsm/ltsm_hf/scripts/' + 'trues.npy', metrics['trues'])
        trainer.log_metrics("Test", metrics)
        trainer.save_metrics("Test", metrics)


if __name__ == "__main__":
    args = get_args()
    seed_all(args.seed)
    run(args)
