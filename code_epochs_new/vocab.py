import json
import os
import sys
import re
import numpy as np
import pickle
import spacy
from spacy.tokenizer import Tokenizer
from argparse import ArgumentParser, Namespace
import stanza
from tqdm import tqdm


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
    parser.add_argument("--spacy", type=str, default="no")
    parser.add_argument("--stanza", type=str, default="no")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # CONFIG_NAME = 'config.json'
    args = get_args()
    CONFIG_NAME = args.cnfg_nm
    BASE_PATH, CONFIG = read_config(CONFIG_NAME)
    with open(CONFIG["DATA"]["DATA_PATH"]) as f:
        data_tr = json.load(f)
    with open(CONFIG["DATA"]["VAL_PATH"]) as f:
        data_val = json.load(f)
    with open(CONFIG["DATA"]["TEST_PATH"]) as f:
        data_test = json.load(f)
    ## use only Polarity-Intensity tags in the B and S tags for Polar expressions.
    if "_eu" not in CONFIG_NAME:
        pos_tagger = spacy.load(CONFIG["MODEL"]["SPACY_MODEL_PATH"])
        pos_tagger.tokenizer = Tokenizer(pos_tagger.vocab, token_match=re.compile(r'\S+').match)
    def_lang = "en"
    if "LANG" in CONFIG["DATA"]:
        def_lang = CONFIG["DATA"]["LANG"]
    stanza_pipe = stanza.Pipeline(lang=def_lang, processors='tokenize,pos,lemma,depparse',
                                  dir=CONFIG["MODEL"]["STANZA_MODEL_PATH"], tokenize_no_ssplit=True, tokenize_pretokenized=True)
    data = [i["text"] for i in data_tr] + [i["text"] for i in data_val] + [i["text"] for i in data_test]
    stanza_lemma_vocab = {}
    stanza_upos_vocab = {}
    stanza_xpos_vocab = {}
    stanza_depparse_vocab = {}

    spacy_lemma_vocab = {}
    spacy_upos_vocab = {}
    spacy_xpos_vocab = {}
    spacy_depparse_vocab = {}

    for i in tqdm(data, total=len(data)):
        doc = stanza_pipe(i)
        for sentence in doc.sentences:
            for token in sentence.words:
                if token.lemma not in stanza_lemma_vocab:
                    stanza_lemma_vocab[token.lemma] = len(stanza_lemma_vocab)
                if token.upos not in stanza_upos_vocab:
                    stanza_upos_vocab[token.upos] = len(stanza_upos_vocab)
                if token.xpos not in stanza_xpos_vocab:
                    stanza_xpos_vocab[token.xpos] = len(stanza_xpos_vocab)
                if token.deprel not in stanza_depparse_vocab:
                    stanza_depparse_vocab[token.deprel] = len(stanza_depparse_vocab)
        if "_eu" not in CONFIG_NAME:
            for token in pos_tagger(i):
                if token.pos_ not in spacy_upos_vocab:
                    spacy_upos_vocab[token.pos_] = len(spacy_upos_vocab)
                if token.tag_ not in spacy_xpos_vocab:
                    spacy_xpos_vocab[token.tag_] = len(spacy_xpos_vocab)
                if token.lemma_ not in spacy_lemma_vocab:
                    spacy_lemma_vocab[token.lemma_] = len(spacy_lemma_vocab)
                if token.dep_ not in spacy_depparse_vocab:
                    spacy_depparse_vocab[token.dep_] = len(spacy_depparse_vocab)

    print("stanza_lemma_vocab::{0}, stanza_upos_vocab::{1}, stanza_xpos_vocab::{2},"
          " stanza_depparse_vocab::{3}".format(len(stanza_lemma_vocab),len(stanza_upos_vocab),
                                              len(stanza_xpos_vocab),len(stanza_depparse_vocab)))
    print("spacy_lemma_vocab::{0}, spacy_upos_vocab::{1}, spacy_xpos_vocab::{2},"
          " spacy_depparse_vocab::{3}".format(len(spacy_lemma_vocab), len(spacy_upos_vocab),
                                              len(spacy_xpos_vocab), len(spacy_depparse_vocab)))
    print("saving all vocabs")
    with open(CONFIG["DATA"]["SAVE_VOCAB"]+'stanza_lemma_vocab.json', 'w') as fp:
        json.dump(stanza_lemma_vocab, fp)
    with open(CONFIG["DATA"]["SAVE_VOCAB"]+'stanza_upos_vocab.json', 'w') as fp:
        json.dump(stanza_upos_vocab, fp)
    with open(CONFIG["DATA"]["SAVE_VOCAB"]+'stanza_xpos_vocab.json', 'w') as fp:
        json.dump(stanza_xpos_vocab, fp)
    with open(CONFIG["DATA"]["SAVE_VOCAB"]+'stanza_depparse_vocab.json', 'w') as fp:
        json.dump(stanza_depparse_vocab, fp)
    if "_eu" not in CONFIG_NAME:
        with open(CONFIG["DATA"]["SAVE_VOCAB"]+'spacy_lemma_vocab.json', 'w') as fp:
            json.dump(spacy_lemma_vocab, fp)
        with open(CONFIG["DATA"]["SAVE_VOCAB"]+'spacy_upos_vocab.json', 'w') as fp:
            json.dump(spacy_upos_vocab, fp)
        with open(CONFIG["DATA"]["SAVE_VOCAB"]+'spacy_xpos_vocab.json', 'w') as fp:
            json.dump(spacy_xpos_vocab, fp)
        with open(CONFIG["DATA"]["SAVE_VOCAB"]+'spacy_depparse_vocab.json', 'w') as fp:
            json.dump(spacy_depparse_vocab, fp)
