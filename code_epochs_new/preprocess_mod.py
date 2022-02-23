import json
import os
import sys
import re
import torch
import numpy as np
import pickle
import transformers
from tqdm import tqdm
import spacy
import stanza
from spacy.tokenizer import Tokenizer

class EntityDataset:
    def read_config(self,config_file):
        # abspath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        basepath = os.path.join(basepath, "config")
        with open(os.path.join(basepath, config_file), 'r') as f:
            config = json.load(f)
        return basepath, config

    def clean_text(self, sent):
        # return re.sub(r'[^\w\s]', '', sent).lower()
        return sent.lower()

    def BIO_tags(self, tokenised, tags, sent):
        adj_mat = np.zeros((len(self.unq_lab),len(tokenised),len(tokenised)), dtype=int)
        def get_tok_span(sent,start_off,end_off):
            begin_tok_no = 0
            if start_off > 0:
                begin_part, _, _, _, _ = self.tokenize(self.clean_text(sent[0:start_off]))
                begin_tok_no = len(begin_part)
            mid_part, _, _, _, _ = self.tokenize(self.clean_text(sent[start_off:end_off]))
            mid_tok_no = len(mid_part)
            return begin_tok_no,mid_tok_no
        def excep_handle_ones(adj_mat,layer1,layer2,layer3_st,layer3_end):
            # print(layer1,layer2,layer3_st,layer3_end)
            try:
                if layer3_end == -1000:
                    adj_mat[layer1][layer2][layer3_st] = 1
                else:
                    adj_mat[layer1][layer2][layer3_st:layer3_end] = 1
                return adj_mat
            except IndexError:
                return adj_mat
        if len(tags) > 0:
            for tag_one in tags:
                if tag_one["Polarity"]:
                    key_exps = []
                    try:
                        if len(tag_one["Polar_expression"][0]) > 0:
                            for mult in tag_one["Polar_expression"][1]:
                                key_exp = get_tok_span(sent,int(mult.split(':')[0].strip()),
                                                         int(mult.split(':')[1].strip()))
                                key_exps.append(key_exp)
                                if tag_one["Polarity"] == "Positive":
                                    adj_mat = excep_handle_ones(adj_mat, 1, key_exp[0], key_exp[0], key_exp[0]+key_exp[1])
                                elif tag_one["Polarity"] == "Negative":
                                    adj_mat = excep_handle_ones(adj_mat, 2, key_exp[0], key_exp[0], key_exp[0] + key_exp[1])
                        if len(tag_one["Source"][0]) > 0:
                            for mult in tag_one["Source"][1]:
                                key_src = get_tok_span(sent, int(mult.split(':')[0].strip()),
                                       int(mult.split(':')[1].strip()))
                                adj_mat = excep_handle_ones(adj_mat, 4, key_src[0], key_src[0], key_src[0] + key_src[1])
                                for key_exp in key_exps:
                                    if tag_one["Polarity"] == "Positive" and key_exp:
                                        adj_mat = excep_handle_ones(adj_mat, 1, key_exp[0], key_src[0], -1000)
                                    elif tag_one["Polarity"] == "Negative" and key_exp:
                                        adj_mat = excep_handle_ones(adj_mat, 2, key_exp[0], key_src[0], -1000)
                        if len(tag_one["Target"][0]) > 0:
                            for mult in tag_one["Target"][1]:
                                key_tar = get_tok_span(sent, int(mult.split(':')[0].strip()),
                                       int(mult.split(':')[1].strip()))
                                adj_mat = excep_handle_ones(adj_mat, 3, key_tar[0], key_tar[0], key_tar[0] + key_tar[1])
                                for key_exp in key_exps:
                                    if tag_one["Polarity"] == "Positive" and key_exp:
                                        adj_mat = excep_handle_ones(adj_mat, 1, key_exp[0], key_tar[0], -1000)
                                    elif tag_one["Polarity"] == "Negative" and key_exp:
                                        # adj_mat[2][key_exp[0]][key_tar[0]] = 1
                                        adj_mat = excep_handle_ones(adj_mat, 2, key_exp[0], key_tar[0], -1000)
                    except IndexError:
                        print(tags)
                        print(sent)
                        print(tokenised)
                        print(key_exp)
        else:
            # if no tags present return "O" only
            return adj_mat

        return adj_mat

    # the CLS,SEP and padded words = -100
    def encode_labels(self, label):
        label = label[:self.PAD_MAX_LENGTH]
        label = label + ['PAD'] * (self.PAD_MAX_LENGTH - len(label))
        label_en = np.zeros((self.PAD_MAX_LENGTH,len(self.unq_lab)), dtype=int)
        for no,i in enumerate(label):
            label_en[no,self.label_map[i]]=1
        return label_en

    def tokenize(self, sent, return_offsets = False):
        encoded_dict = self.tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # don't add '[CLS]' and '[SEP]'
            max_length=self.PAD_MAX_LENGTH,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        tokenised = self.tokenizer.tokenize(sent)
        token_offsets = []
        tok_cnt = []
        if return_offsets:
            space_tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r'\S+', sent)]
            space_sub_tokens =[]
            for i in space_tokens:
                for j in self.tokenizer.tokenize(i[0]):
                    space_sub_tokens.append((i[1],i[2]))
                tok_cnt.append(len(self.tokenizer.tokenize(i[0])))
            token_offsets = space_sub_tokens
        # Add the encoded sentence to the list.
        input_ids = encoded_dict['input_ids']
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks = encoded_dict['attention_mask']
        return tokenised, input_ids, attention_masks, token_offsets, tok_cnt

    def get_spacy_pos(self, txt, tok_cnts):
        ref_pos = {"ADJ":0, "ADP":1, "ADV":2, "AUX":3, "CCONJ":4, "DET":5, "INTJ":6, "NOUN":7, "NUM":8, "PART":9,
                   "PRON":10, "PROPN":11, "PUNCT":12, "SCONJ":13, "SYM":14, "VERB":15, "X":16}
        pos = []
        for token,span in zip(self.pos_tagger(txt),tok_cnts):
            temp = [[0.0]*17]
            temp[0][ref_pos[token.pos_]] = 1.0
            pos = temp*span + (self.PAD_MAX_LENGTH-span)*[[0.0]*17]
        return pos

    def get_vec(self,toks,file_path,tok_cnts):
        with open(file_path, 'r') as f:
            ref = json.load(f)
        pos = []
        for token,span in zip(toks,tok_cnts):
            temp = [[0.0]*len(ref)]
            temp[0][ref[token]] = 1.0
            pos += temp*span
        if len(pos) < self.PAD_MAX_LENGTH:
            pos += (self.PAD_MAX_LENGTH - len(pos)) * [[0.0] * len(ref)]
        elif len(pos) > self.PAD_MAX_LENGTH:
            pos = pos[:self.PAD_MAX_LENGTH]
        pos = np.array(pos)
        return pos

    def get_context_vector(self, txt, tok_cnts, pos_tagger, stanza_pipe):
        stanza_out = stanza_pipe(txt)
        spacy_out = pos_tagger(txt)
        context = {"stanza_lemma_vocab":[],"stanza_upos_vocab":[],"stanza_xpos_vocab":[],"stanza_depparse_vocab":[],
                   "spacy_lemma_vocab":[],"spacy_upos_vocab":[],"spacy_xpos_vocab":[],"spacy_depparse_vocab":[]}
        for sentence in stanza_out.sentences:
            for token in sentence.words:
                if self.cntx_options["stanza_lemma_vocab"] == "yes":
                    context["stanza_lemma_vocab"].append(token.lemma)
                if self.cntx_options["stanza_upos_vocab"] == "yes":
                    context["stanza_upos_vocab"].append(token.upos)
                if self.cntx_options["stanza_xpos_vocab"] == "yes":
                    context["stanza_xpos_vocab"].append(token.xpos)
                if self.cntx_options["stanza_depparse_vocab"] == "yes":
                    context["stanza_depparse_vocab"].append(token.deprel)
        for token in spacy_out:
            if self.cntx_options["spacy_lemma_vocab"] == "yes":
                context["spacy_lemma_vocab"].append(token.lemma_)
            if self.cntx_options["spacy_upos_vocab"] == "yes":
                context["spacy_upos_vocab"].append(token.pos_)
            if self.cntx_options["spacy_xpos_vocab"] == "yes":
                context["spacy_xpos_vocab"].append(token.tag_)
            if self.cntx_options["spacy_depparse_vocab"] == "yes":
                context["spacy_depparse_vocab"].append(token.dep_)
        pos = []
        for i in self.cntx_options:
            if self.cntx_options[i] == "yes":
                pos.append(self.get_vec(context[i], self.config["DATA"]["SAVE_VOCAB"]+i+".json", tok_cnts))
        pos = np.concatenate(pos, axis=1)
        return pos

    def prepare_data(self, sent, tags, opt):
        sent_pr = self.clean_text(sent)
        tokenised, input_id, attention_mask, token_offsets, tok_cnt = self.tokenize(sent_pr, return_offsets=True)
        if opt == "train" or opt == "valid":
            adj_mat = self.BIO_tags(tokenised, tags, sent)
            return input_id, attention_mask, adj_mat, token_offsets, tok_cnt
        return input_id, attention_mask, token_offsets, tok_cnt

    def __init__(self, opt, CONFIG_NAME):
        # CONFIG_NAME = 'config.json'
        BASE_PATH, CONFIG = self.read_config(CONFIG_NAME)
        if opt == "train":
            with open(CONFIG["DATA"]["DATA_PATH"]) as f:
                self.data = json.load(f)
        elif opt == "valid":
            with open(CONFIG["DATA"]["VAL_PATH"]) as f:
                self.data = json.load(f)
        else:
            with open(CONFIG["DATA"]["TEST_PATH"]) as f:
                self.data = json.load(f)
        self.config = CONFIG
        self.unq_lab = ['O', 'EXP-POS', 'EXP-NEG', 'TAR', 'SRC']
        self.class_ref = {'Negative':'NEG','Positive':'POS'}
        self.label_rev_map = {i: lab for i, lab in enumerate(self.unq_lab)}
        self.label_map = {lab: i for i, lab in enumerate(self.unq_lab)}
        self.MODEL_PATH = CONFIG["MODEL"]["MODEL_PATH"]
        self.PAD_MAX_LENGTH = CONFIG["MODEL"]["PAD_MAX_LENGTH"]
        self.tokenizer = transformers.BertTokenizer.from_pretrained(os.path.join(self.MODEL_PATH,'vocab.txt'), do_lower_case=True)
        print("Loaded tokenizer here!!")

        # data after BIO tagging and POS tags from stanza..
        self.sents = []
        self.input_ids = []
        self.attention_masks = []
        self.adj_mats = []
        self.token_offsets_all = []
        self.opinions = []
        self.sent_ids = []
        self.sent_ids_ = []
        self.sents_ = []
        self.context_vec = []
        self.tok_cnts = []
        for i in self.data:
            # if i['text'] and len(i['text'].strip()) > 0 and len(i['opinions']) > 0:
            if i['text'] and len(i['text'].strip()) > 0 and len(re.sub(r'[^a-zA-Z]', '', i['text'])) > 0:
                if opt == "train" or opt == "valid":
                    input_id, attention_mask, adj_mat, token_offsets, tok_cnt = self.prepare_data(i['text'], i['opinions'], opt)
                else:
                    input_id, attention_mask, token_offsets, tok_cnt = self.prepare_data(i['text'], i['opinions'], opt)
                self.token_offsets_all.append(token_offsets)
                self.sents.append(i['text'])
                self.opinions.append(i['opinions'])
                self.sent_ids.append(i['sent_id'])
                self.tok_cnts.append(tok_cnt)
                self.input_ids.append(input_id)
                self.attention_masks.append(attention_mask)
                if opt == "train" or opt == "valid":
                    adj_mat = torch.nn.functional.pad(input=torch.tensor(adj_mat), pad=(0,self.PAD_MAX_LENGTH-adj_mat.shape[1],
                                                                                        0,self.PAD_MAX_LENGTH-adj_mat.shape[2],0,0),
                                                      mode='constant', value=-100).tolist()
                    self.adj_mats.append(adj_mat)
            else:
                self.sent_ids_.append(i['sent_id'])
                self.sents_.append(i['text'])

        if "CONTEXT" in CONFIG:
            pos_tagger = spacy.load(self.config["MODEL"]["SPACY_MODEL_PATH"])
            pos_tagger.tokenizer = Tokenizer(pos_tagger.vocab, token_match=re.compile(r'\S+').match)
            def_lang = "en"
            if "LANG" in CONFIG["DATA"]:
                def_lang = CONFIG["DATA"]["LANG"]
            stanza_pipe = stanza.Pipeline(lang=def_lang, processors='tokenize,pos,lemma,depparse',
                                          dir=self.config["MODEL"]["STANZA_MODEL_PATH"], tokenize_no_ssplit=True,
                                          tokenize_pretokenized=True)
            self.cntx_options = CONFIG["CONTEXT"]
            for i,j in tqdm(zip(self.sents,self.tok_cnts), total=len(self.sents)):
                self.context_vec.append(self.get_context_vector(i, j, pos_tagger, stanza_pipe))
            # self.context_vec = np.array(self.context_vec).astype(np.float32)
            self.context_vec = torch.tensor(self.context_vec, dtype=torch.float32)
            print("CONTEXT VECTOR SHAPE:::",self.context_vec.shape)
        # print(self.pos_tags)
        # Convert the lists into tensors.
        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.adj_mats = torch.tensor(self.adj_mats)
        print("PREPROCESSING COMPLETE!!")

    def __len__(self):
        return len(self.data)