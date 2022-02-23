import json
import os
import sys
import re
import numpy as np
import pickle
import warnings
from argparse import ArgumentParser, Namespace
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

def get_args():
    parser = ArgumentParser(
        description="Options for fine-tuning or evaluation of BERT-BASE-BL model")
    parser.add_argument("--cnfg_nm", type=str, default="config.json")
    parser.add_argument("--dir", type=str, default="./")
    parser.add_argument("--opt", type=str, default="test")
    args = parser.parse_args()
    return args

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

def post_process(data):
    for no,i in enumerate(data):
        opinions = i["opinions"]
        opinions_post = []
        for j in opinions:
            source_inst = j["Source"][0]
            source_offs = j["Source"][1]
            target_inst = j["Target"][0]
            target_offs = j["Target"][1]
            polarity = j["Polarity"]
            Intensity = j["Intensity"]
            temp = []
            for src,offs1 in zip(source_inst,source_offs):
                for tar,offs2 in zip(target_inst,target_offs):
                    temp.append({"Polar_expression":j["Polar_expression"].copy(),
                                 "Source":[[src],[offs1]],
                                 "Target": [[tar],[offs2]],
                                 "Polarity": polarity,
                                 "Intensity": Intensity
                                 })
            if len(source_inst)==0:
                for tar,offs2 in zip(target_inst,target_offs):
                    temp.append({"Polar_expression":j["Polar_expression"].copy(),
                                 "Source":[[],[]],
                                 "Target": [[tar],[offs2]],
                                 "Polarity": polarity,
                                 "Intensity": Intensity
                                 })
            if len(target_inst)==0:
                for src,offs1 in zip(source_inst,source_offs):
                    temp.append({"Polar_expression":j["Polar_expression"].copy(),
                                 "Source":[[src],[offs1]],
                                 "Target": [[],[]],
                                 "Polarity": polarity,
                                 "Intensity": Intensity
                                 })
            if len(source_inst) == 0 and len(target_inst)==0:
                temp.append({"Polar_expression":j["Polar_expression"].copy(),
                             "Source":[[],[]],
                             "Target": [[],[]],
                             "Polarity": polarity,
                             "Intensity": Intensity
                             })
            opinions_post.extend(temp)
        data[no]["opinions"] = opinions_post
    return data

def remove_tar_src_frm_exp(tup,sent):
    exp = [int(i) for i in tup["Polar_expression"][1][0].split(':')]
    src = tup["Source"]
    tar = tup["Target"]
    if len(tar[0]) > 0:
        tar = [int(i) for i in tar[1][0].split(':')]
        if tar[0] == exp[0] and (tar[1] == exp[1] or tar[1] == exp[1]-1):
            return {}
        elif tar[1] == exp[1] and tar[0]-1 == exp[0]:
            return {}
        elif tar[0] <= exp[0] and tar[1] < exp[1] and tar[1] > exp[0]:
            exp_st = tar[1]+1
            exp_end = exp[1]
            tup["Polar_expression"] = [[sent[exp_st:exp_end]], [str(exp_st)+':'+str(exp_end)]]
        elif tar[1] >= exp[1] and tar[0] > exp[0] and tar[0] < exp[1]:
            exp_st = exp[0]
            exp_end = tar[0]-1
            tup["Polar_expression"] = [[sent[exp_st:exp_end]], [str(exp_st) + ':' + str(exp_end)]]
    if len(src[0]) > 0:
        src = [int(i) for i in src[1][0].split(':')]
        if src[0] == exp[0] and (src[1] == exp[1] or src[1] == exp[1] - 1):
            return {}
        elif src[1] == exp[1] and src[0] - 1 == exp[0]:
            return {}
        elif src[0] <= exp[0] and src[1] < exp[1] and src[1] > exp[0]:
            exp_st = src[1] + 1
            exp_end = exp[1]
            tup["Polar_expression"] = [[sent[exp_st:exp_end]], [str(exp_st) + ':' + str(exp_end)]]
        elif src[1] >= exp[1] and src[0] > exp[0] and src[0] < exp[1]:
            exp_st = exp[0]
            exp_end = src[0] - 1
            tup["Polar_expression"] = [[sent[exp_st:exp_end]], [str(exp_st) + ':' + str(exp_end)]]
    return tup

def remove_overlaps(opinions, sent):
    res = []
    exp_pool = list(set([i["Polar_expression"][0][0] for i in opinions if len(i["Polar_expression"][0])>0]))
    for i in opinions:
        # if not any((i["Polar_expression"][0][0] in substring
        #         and i["Polar_expression"][0][0] != substring) for substring in exp_pool):
        if len(i["Polar_expression"][0]) > 0:
            temp = remove_tar_src_frm_exp(i, sent)
            if len(temp) > 0:
                res.append(temp)
    return res

if __name__ == "__main__":
    args = get_args()
    CONFIG_NAME = args.cnfg_nm
    _, CONFIG = read_config(CONFIG_NAME)
    with open(CONFIG["DATA"]["SAVE_PATH"]) as f:
        data = json.load(f)

    data = post_process(data)
    results = []
    for i in data:
        temp = i.copy()
        opinions = remove_overlaps(i["opinions"], i["text"])
        temp["opinions"] = opinions
        results.append(temp)

    with open("/axp/rim/bdlml/dev/psarangi/semeval2020/post_processed/monolingual/"
              +CONFIG["DATA"]["SAVE_PATH"].split('/')[-2]+"/predictions.json", 'w') as fp:
        json.dump(results, fp)