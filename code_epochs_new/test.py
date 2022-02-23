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
sys.path.insert(1, '/axp/rim/bdlml/dev/psarangi/semeval2020/evaluate/')
from evaluate_single_dataset import main

warnings.filterwarnings("ignore")
def get_last(row, last, ind):
    off = last
    for no, i in enumerate(row):
        if no == 0:
            continue
        elif i != ind:
            break
        elif i == ind:
            off = last + no
    return off

def read_config(config_file):
    # abspath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    basepath = os.path.join(basepath, "config")
    with open(os.path.join(basepath, config_file), 'r') as f:
        config = json.load(f)
    return basepath, config

def get_args():
    parser = ArgumentParser(
        description="Options for fine-tuning or evaluation of BERT-BASE-BL model")
    parser.add_argument("--cnfg_nm", type=str, default="config.json")
    parser.add_argument("--dir", type=str, default="./")
    parser.add_argument("--opt", type=str, default="test")
    args = parser.parse_args()
    return args

def get_tokens_from_preds(temp):
    tar_offs = {}
    src_offs = {}
    for i in range(len(temp)):
        start_off = i
        row = list(temp[i])
        if temp[i][i] == 3:
            end_off = get_last(row[i:], i, temp[i][i])
            tar_offs[i] = (start_off, end_off)
        elif temp[i][i] == 4:
            end_off = get_last(row[i:], i, temp[i][i])
            src_offs[i] = (start_off, end_off)
    arcs = {}
    for i in range(len(temp)):
        if temp[i][i] == 1 or temp[i][i] == 2:
            arcs[i] = {"Source": [], "Target": [], "Polar_expression":[], "Polarity":"Negative", "Intensity": "Standard"}
            row = list(temp[i])
            exp_st_off = i
            exp_end_off = get_last(row[i:], i, temp[i][i])
            row = list(temp[i])
            for j in range(len(row)):
                if j != i and row[j] == temp[i][i]:
                    if j in tar_offs:
                        arcs[i]["Target"].append(tar_offs[j])
                    elif j in src_offs:
                        arcs[i]["Source"].append(src_offs[j])
            if temp[i][i] == 1:
                arcs[i]["Polar_expression"].append((exp_st_off,exp_end_off))
                arcs[i]["Polarity"] = "Positive"
            else:
                arcs[i]["Polar_expression"].append((exp_st_off, exp_end_off))
                arcs[i]["Polarity"] = "Negative"

    arcs = list(arcs.values())
    # opinions_tar_src = []
    # included = []
    # for no1,i in enumerate(arcs):
    #     temp = i.copy()
    #     for no2,j in enumerate(arcs):
    #         if no2 != no1 and no1 not in included and i["Source"] == j["Source"] \
    #                 and i["Target"] == j["Target"] and i["Polarity"] == j["Polarity"]:
    #             included.append(no2)
    #             temp["Polar_expression"].append(j["Polar_expression"][0])
    #     if no1 not in included:
    #         opinions_tar_src.append(temp)
    return arcs

def remove_tar_src_frm_exp(tup,sent):
    # exp = [int(i) for i in tup["Polar_expression"][1][0].split(':')]
    src = [[int(x) for x in i.split(':')] for i in tup["Source"][1]]
    tar = [[int(x) for x in i.split(':')] for i in tup["Target"][1]]
    exps1 = []
    exps2 = []
    for i,j in zip(tup["Polar_expression"][0],tup["Polar_expression"][1]):
        exps1.append(i)
        exps2.append(j)
        st = int(j.split(':')[0])
        end = int(j.split(':')[1])
        for [m,n] in src:
            if m==st and n<end:
                exps1[-1] = sent[n:end]
                exps2[-1] = str(n)+":"+str(end)
            elif m>st and n==end:
                exps1[-1] = sent[st:m]
                exps2[-1] = str(st)+":"+str(m)
            elif m<=st and n>=end:
                exps1[-1] = ""
                exps2[-1] = ""
        if exps1[-1]!="":
            for [m,n] in tar:
                if m==st and n<end:
                    exps1[-1] = sent[n:end]
                    exps2[-1] = str(n)+":"+str(end)
                elif m>st and n==end:
                    exps1[-1] = sent[st:m]
                    exps2[-1] = str(st)+":"+str(m)
                elif m<=st and n>=end:
                    exps1[-1] = ""
                    exps2[-1] = ""
        if exps1[-1]=="":
            exps1.pop(-1)
            exps2.pop(-1)
    tup["Polar_expression"] = [exps1,exps2]
    return tup

def post_process(opinions, sent):
    res = []
    # exp_pool = list(set([i["Polar_expression"][0][0] for i in opinions if len(i["Polar_expression"][0])>0]))
    for i in opinions:
        # if not any((i["Polar_expression"][0][0] in substring
        #         and i["Polar_expression"][0][0] != substring) for substring in exp_pool):
        temp = remove_tar_src_frm_exp(i.copy(), sent)
        if len(temp) > 0:
            res.append(temp)
    return res
def get_tuples(test_preds, padded_lens, test_dataset_obj):
    results = []
    results_post = []
    for i, j, l, m, n, o in zip(test_preds, padded_lens, test_dataset_obj.token_offsets_all, test_dataset_obj.sents
            , test_dataset_obj.opinions, test_dataset_obj.sent_ids):
        result = {"sent_id": o, "text": m}
        result_post = {"sent_id": o, "text": m}
        temp = np.array(i[:, :j, :j]).argmax(axis=0)
        # temp = np.array(k[:, :j, :j]).argmax(axis=0)
        opinions = get_tokens_from_preds(temp)
        # print(n)
        for no, ops in enumerate(opinions):
            if len(opinions[no]["Source"]) > 0:
                temp1 = []
                temp2 = []
                for no1,temp in enumerate(opinions[no]["Source"]):
                    st_off = l[temp[0]][0]
                    end_off = l[temp[1]][1]
                    temp1.append(m[st_off:end_off])
                    temp2.append(str(st_off) + ':' + str(end_off))
                opinions[no]["Source"] = [temp1,temp2]
            else:
                opinions[no]["Source"] = [[],[]]
            if len(opinions[no]["Target"]) > 0:
                temp1 = []
                temp2 = []
                for no1, temp in enumerate(opinions[no]["Target"]):
                    st_off = l[temp[0]][0]
                    end_off = l[temp[1]][1]
                    temp1.append(m[st_off:end_off])
                    temp2.append(str(st_off) + ':' + str(end_off))
                opinions[no]["Target"] = [temp1, temp2]
            else:
                opinions[no]["Target"] = [[],[]]
            if len(opinions[no]["Polar_expression"]) > 0:
                temp1 = []
                temp2 = []
                for no1, temp in enumerate(opinions[no]["Polar_expression"]):
                    st_off = l[temp[0]][0]
                    end_off = l[temp[1]][1]
                    temp1.append(m[st_off:end_off])
                    temp2.append(str(st_off) + ':' + str(end_off))
                opinions[no]["Polar_expression"] = [temp1, temp2]
        result["opinions"] = opinions
        results.append(result)
        ## postprocessor added
        result_post["opinions"] = post_process(opinions, m)
        results_post.append(result_post)
        ## postprocessor added
    for i, j in zip(test_dataset_obj.sent_ids_, test_dataset_obj.sents_):
        results.append({"sent_id": i, "text": j, "opinions": []})
        results_post.append({"sent_id": i, "text": j, "opinions": []})
    return results_post

if __name__ == "__main__":
    print("START TESTING!!!")
    args = get_args()
    CONFIG_NAME = args.cnfg_nm
    _, CONFIG = read_config(CONFIG_NAME)
    test_dataset_obj = EntityDataset("test",CONFIG_NAME)
    # test_dataset_obj = EntityDataset("train",CONFIG_NAME)
    padded_lens = []
    for i in test_dataset_obj.attention_masks:
        padded_lens.append(int(sum(i)))
    test_dataset = TensorDataset(test_dataset_obj.input_ids, test_dataset_obj.attention_masks, test_dataset_obj.context_vec)
    test_dataloader = DataLoader(
        test_dataset, # The test samples.
        batch_size=1 # pass one test sample at a time
    )
    print("MODEL EPOCH RUN :: " + str(CONFIG["MODEL"]["MODEL_TEST_PATH"][-2:]))
    model = torch.load(CONFIG["MODEL"]["MODEL_TEST_PATH"])
    devices = "cuda:"
    for i in CONFIG["MODEL"]["GPUS_USED"]:
        devices += str(i) + ","
    device = torch.device(devices if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("We have ", torch.cuda.device_count(), " GPUs!")
        model_parallel = torch.nn.DataParallel(model, device_ids=CONFIG["MODEL"]["GPUS_USED"])
    model_parallel.to(device)
    test_preds = engine.test_fn(test_dataloader, model_parallel, device, "test")
    results = get_tuples(test_preds, padded_lens, test_dataset_obj)

    with open(CONFIG["DATA"]["SAVE_PATH"], 'w') as fp:
        json.dump(results, fp)

    # for i in range(17,18):
    #     print("MODEL EPOCH RUN :: "+str(i+1))
    #     model = torch.load(CONFIG["MODEL"]["MODEL_TEST_PATH"][:-2]+str(i+1))
    #     devices = "cuda:"
    #     for i in CONFIG["MODEL"]["GPUS_USED"]:
    #         devices += str(i) + ","
    #     device = torch.device(devices if torch.cuda.is_available() else "cpu")
    #     if torch.cuda.device_count() > 1:
    #         print("We have ", torch.cuda.device_count(), " GPUs!")
    #         model_parallel = torch.nn.DataParallel(model, device_ids=CONFIG["MODEL"]["GPUS_USED"])
    #     model_parallel.to(device)
    #     test_preds = engine.test_fn(test_dataloader, model_parallel, device, "test")
    #     results = get_tuples(test_preds, padded_lens, test_dataset_obj)
    #
    #     with open(CONFIG["DATA"]["SAVE_PATH"], 'w') as fp:
    #         json.dump(results, fp)
    #
    #     main(gold_path=CONFIG["DATA"]["TEST_PATH"],
    #          pred_path=CONFIG["DATA"]["SAVE_PATH"])

        # with open(CONFIG["DATA"]["SAVE_PATH"], 'w') as fp:
        #     json.dump(results_post, fp)
        #
        # main(gold_path=CONFIG["DATA"]["TEST_PATH"],
        #      pred_path=CONFIG["DATA"]["SAVE_PATH"])