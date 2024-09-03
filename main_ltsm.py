import numpy as np
import torch
from torch import nn
import os
import argparse
import random
import ipdb

from ltsm.data_provider.data_factory import get_data_loaders, get_datasets,get_test_datasets
from ltsm.data_provider.data_loader import HF_Dataset
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

    # Basic Config
    parser.add_argument('--model_id', type=str, default='test_run', help='model id')
    parser.add_argument('--model_name_or_path', type=str, default="gpt2-medium", help='model name')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Data Settings
    parser.add_argument('--data_path', nargs='+', default='dataset/weather.csv', help='data files')
    parser.add_argument('--test_data_path_list', nargs='+', required=True, help='test data file')
    parser.add_argument('--prompt_data_path', type=str, default='./weather.csv', help='prompt data file')
    parser.add_argument('--data_processing', type=str, default="standard_scaler", help='data processing method')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation data ratio')

    # Forecasting Settings
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--prompt_len', type=int, default=133, help='prompt sequence length')

    # Model Settings
    parser.add_argument('--lora', action="store_true", help='use lora')
    parser.add_argument('--lora_dim', type=int, default=128, help='dimension of lora')
    parser.add_argument('--gpt_layers', type=int, default=3, help='number of gpt layers')
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='number of heads')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=862, help='output size')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')
    parser.add_argument('--pretrain', type=int, default=1, help='is pretrain')
    parser.add_argument('--local_pretrain', type=str, default="None", help='local pretrain weight')
    parser.add_argument('--freeze', type=int, default=1, help='is model weight frozen')
    parser.add_argument('--model', type=str, default='model', help='model name, , options:[LTSM, LTSM_WordPrompt, LTSM_Tokenizer]')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--tmax', type=int, default=10, help='tmax')
    
    # Training Settings 
    parser.add_argument('--eval', type=int, default=0, help='evaluation')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--output_dir', type=str, default='output/ltsm_train_lr0005/', help='output directory')
    parser.add_argument('--downsample_rate', type=int, default=100, help='downsample rate')
    parser.add_argument('--llm_layers', type=int, default=32)
    parser.add_argument('--decay_fac', type=float, default=0.75, help='decay factor')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
    parser.add_argument('--train_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment type')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64, help='gradient accumulation steps')
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


def run(args):
    print(args)
    
    model_config = LTSMConfig(**vars(args))
    model = get_model(model_config)

    if args.lora:
        peft_config = LoraConfig(
            target_modules=["c_attn"],
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

    # Optimizer settings
    model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    # Evaluation metrics
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        if preds.shape != p.label_ids.shape:
            label_ids = np.squeeze(p.label_ids)
        else:
            label_ids = p.label_ids
        return {
                "mse": ((preds - label_ids) ** 2).mean().item(),
                "mae": (np.abs(preds - label_ids)).mean().item()
        }

    # Loss function
    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(inputs["input_data"])
        loss = nn.functional.mse_loss(outputs, inputs["labels"])
        return (loss, outputs) if return_outputs else loss

    # Data collator
    def collate_fn(batch):
        return {
            'input_data': torch.from_numpy(np.stack([x['input_data'] for x in batch])).type(torch.float32),
            'labels': torch.from_numpy(np.stack([x['labels'] for x in batch])).type(torch.float32),
        }

    # Prediction step
    @torch.no_grad()
    def prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None):
        # CSV
        input_data = inputs["input_data"].to(model.device)
        labels = inputs["labels"].to(model.device)
        outputs = model(input_data)
        loss = nn.functional.mse_loss(outputs, labels)
        return (loss, outputs, labels)

    # Training settings
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
    
    train_dataset, eval_dataset, _ = get_datasets(args)
    train_dataset, eval_dataset= HF_Dataset(train_dataset), HF_Dataset(eval_dataset)
    
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
        test_dataset, _ = get_test_datasets(args)
        test_dataset = HF_Dataset(test_dataset)

        metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics("Test", metrics)
        trainer.save_metrics("Test", metrics)


if __name__ == "__main__":
    args = get_args()
    seed_all(args.seed)
    run(args)
