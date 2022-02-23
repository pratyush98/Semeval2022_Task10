import json
import os
import sys
import re
import torch
import numpy as np
import pickle
from argparse import ArgumentParser, Namespace
import transformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from model import EntityModel
from preprocess_mod import EntityDataset
import random
import engine
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import warnings
from test import get_tuples
sys.path.insert(1, '/axp/rim/bdlml/dev/psarangi/semeval2020/evaluate/')
from evaluate_single_dataset import main

warnings.filterwarnings("ignore")


def read_config(config_file):
    # abspath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    basepath = os.path.join(basepath, "config")
    with open(os.path.join(basepath, config_file), 'r') as f:
        config = json.load(f)
    return basepath, config

def split_dataset(data):
    print(data)

def get_args():
    parser = ArgumentParser(
        description="Options for fine-tuning or evaluation of BERT-BASE-BL model")
    parser.add_argument("--cnfg_nm", type=str, default="config.json")
    parser.add_argument("--dir", type=str, default="./")
    parser.add_argument("--opt", type=str, default="test")
    args = parser.parse_args()
    return args

def form_plots(df_stats,save_path_plot):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (30, 20)
    # Plot the learning curve.
    plt.plot(df_stats['train loss'], 'b-o', label="Training Loss")
    plt.plot(df_stats['valid loss'], 'g-o', label="Validation Loss")
    plt.plot(df_stats['train TPR'], 'r-o', label="Training TPR")
    plt.plot(df_stats['valid TPR'], 'c-o', label="Validation TPR")
    plt.plot(df_stats['train accuracy'], 'm-o', label="Training accuracy")
    plt.plot(df_stats['valid accuracy'], 'y-o', label="Validation accuracy")
    plt.plot(df_stats['train recall'], 'k-o', label="Training recall")
    plt.plot(df_stats['valid recall'], 'm-*', label="Validation recall")

    # Label the plot.
    plt.title("Training & Validation Loss/TPR/accr/recall")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/TPR/accr/recall")
    plt.legend()
    plt.xticks(list(df_stats['epoch']))
    plt.savefig(save_path_plot)

if __name__ == "__main__":
    # CONFIG_NAME = 'config.json'
    args = get_args()
    CONFIG_NAME = args.cnfg_nm
    _, CONFIG = read_config(CONFIG_NAME)

    print("START TRAINING!!!")
    train_dataset_obj = EntityDataset("train",CONFIG_NAME)
    print("train dataset target shape::", train_dataset_obj.adj_mats.shape)
    val_dataset_obj = EntityDataset("valid", CONFIG_NAME)
    print("validation dataset target shape::", val_dataset_obj.adj_mats.shape)
    print("No. of GPU's available: ",torch.cuda.device_count())
    devices = "cuda:"
    for i in CONFIG["MODEL"]["GPUS_USED"]:
        devices += str(i) + ","
    device = torch.device(devices if torch.cuda.is_available() else "cpu")
    print("Total unique tags: ", len(train_dataset_obj.unq_lab))
    model = EntityModel(dim_emb=CONFIG["MODEL"]["DIM_EMB"], label_count=CONFIG["MODEL"]["LABELS_COUNT"],
                        CONFIG_NAME=CONFIG_NAME)
    if torch.cuda.device_count() > 1:
        model_parallel = torch.nn.DataParallel(model, device_ids=CONFIG["MODEL"]["GPUS_USED"])
    model_parallel.to(device)

    ## setting up the data loader part...
    train_dataset = TensorDataset(train_dataset_obj.input_ids, train_dataset_obj.attention_masks, train_dataset_obj.context_vec,
                                  train_dataset_obj.adj_mats)
    val_dataset = TensorDataset(val_dataset_obj.input_ids, val_dataset_obj.attention_masks, val_dataset_obj.context_vec,
                                  val_dataset_obj.adj_mats)
    # Set the seed value all over the place to make this reproducible.
    seed_val = CONFIG["MODEL"]["SEED_VALUE"]
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(val_dataset)))

    train_batch_size = CONFIG["MODEL"]["TRAIN_BATCH_SIZE"]
    val_batch_size = CONFIG["MODEL"]["EVAL_BATCH_SIZE"]

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=train_batch_size  # Trains with this batch size.
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=val_batch_size  # Evaluate with this batch size.
    )
    ## create the testing batch to check the real evaluate
    print("CREATING TESTING DATA WITH VALIDATION SET!!!!")
    test_dataset_obj = EntityDataset("valid", CONFIG_NAME)
    padded_lens = []
    for i in test_dataset_obj.attention_masks:
        padded_lens.append(int(sum(i)))
    test_dataset = TensorDataset(test_dataset_obj.input_ids, test_dataset_obj.attention_masks,
                                 test_dataset_obj.context_vec)
    test_dataloader = DataLoader(
        test_dataset,  # The test samples.
        batch_size=1  # pass one test sample at a time
    )
    ## create the testing batch to check the real evaluate

    param_optimizer = list(model_parallel.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_dataset) / CONFIG["MODEL"]["TRAIN_BATCH_SIZE"] * CONFIG["MODEL"]["EPOCHS"])
    optimizer = AdamW(optimizer_parameters,
                      lr=3e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["MODEL"]["WARMUP_STEPS"],
        num_training_steps=num_train_steps
    )

    best_f1 = 0.0
    best_epoch = 0
    df_stats = pd.DataFrame(columns=['epoch', 'train loss', 'valid loss', 'train TPR', 'valid TPR', 'train recall',
                                     'valid recall', 'train accuracy', 'valid accuracy'])
    for epoch in range(CONFIG["MODEL"]["EPOCHS"]):
        print("STARTING EPOCH NO.{0}".format(epoch + 1))
        train_loss,train_acc,tpr_train_acc,recall_train = engine.train_fn(train_dataloader, model_parallel, optimizer, device, scheduler, "train")
        val_loss,val_acc,tpr_val_acc,recall_val = engine.eval_fn(validation_dataloader, model_parallel, device, "train")
        train_f1 = (recall_train * tpr_train_acc * 2) / (tpr_train_acc + recall_train + 0.000000000001)
        val_f1 = (recall_val*tpr_val_acc*2)/(tpr_val_acc+recall_val + 0.000000000001)
        print("complete for EPOCH no.{0} training/val".format(epoch+1))
        print("Train Loss = {0} and Valid Loss = {1}".format(train_loss, val_loss))
        print("Train accuracy = {0} and Valid Accuracy = {1}".format(train_acc, val_acc))
        print("Train TPR = {0} and Valid TPR = {1}".format(tpr_train_acc, tpr_val_acc))
        print("Train recall = {0} and Valid recall = {1}".format(recall_train, recall_val))
        print("Train F1 = {0} and Valid F1 = {1}".format(train_f1,val_f1))

        df_stats.loc[epoch] = [epoch+1, train_loss, val_loss, tpr_train_acc, tpr_val_acc,
                               recall_train, recall_val, train_acc, val_acc]
        ## running the evaluation
        # print(val_dataset.input_ids)
        # print(val_dataset.input_ids.shape)
        test_preds = engine.test_fn(test_dataloader, model_parallel, device, "test")
        results = get_tuples(test_preds, padded_lens, test_dataset_obj)
        with open(CONFIG["DATA"]["SAVE_PATH"], 'w') as fp:
            json.dump(results, fp)
        # with open(CONFIG["DATA"]["SAVE_PATH"][:-5] + ".json", 'w') as fp:
        #     json.dump(results_post, fp)
        main(gold_path=CONFIG["DATA"]["VAL_PATH"],
             pred_path=CONFIG["DATA"]["SAVE_PATH"])
        # main(gold_path=CONFIG["DATA"]["TEST_PATH"],
        #      pred_path=CONFIG["DATA"]["SAVE_PATH"][:-5] + ".json")
        ## running the evaluation

        if best_f1 < train_f1:
            best_f1 = train_f1
            best_epoch = epoch+1

        if (epoch+1) > 7:
            model_dir = CONFIG["MODEL"]["MODEL_SAVE_PATH"]
            check = os.path.isdir(model_dir)
            # If folder doesn't exist, then create it.
            if not check:
                os.makedirs(model_dir)
                print("created folder : ", model_dir)
            torch.save(model_parallel, model_dir+"e"+str(epoch+1))
            print("Saved model at EPOCH::{0}".format(epoch + 1))
        print("\n")
    print("BEST TRAINING EPOCH NO.{0}, BEST TRAINING F1: {1}".format(best_epoch, best_f1))
    form_plots(df_stats,CONFIG["DATA"]["SAVE_PLOT"])

