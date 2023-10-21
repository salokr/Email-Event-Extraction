from torch.utils.data import Dataset, DataLoader
from glob import glob
import os, json, copy
import torch
from functools import reduce
import numpy as np
from tqdm import tqdm
from itertools import chain
import itertools, re, ast, math

class seq_tagger_arg_multi_class(Dataset):
    def append_tags(self, sentence, trigger, label):
        indices = trigger['indices'].split(" ")
        new_sen = sentence
        new_lbl = label
        #print(indices)
        try:
            indices.remove("")
        except:
            pass
        if(len(indices)>0):
            indices = [int(idx) for idx in indices]
            i = 1
            prev_i = 0
            new_sen = sentence[:indices[0]] + ['[TRG]']
            new_lbl = label[:indices[0]] + ["O"]
            while(i<len(indices)):
                if(not indices[i]-1 == indices[i-1]):
                    new_sen += sentence[indices[prev_i]: indices[i-1] + 1] + ['[/TRG]'] + sentence[indices[i-1] + 1: indices[i]] + ['[TRG]']
                    new_lbl += label[indices[prev_i]:indices[i-1] + 1] + ["O"] + label[indices[i-1] + 1: indices[i]] + ['O']
                    prev_i = i
                i += 1
            new_sen += sentence[indices[prev_i]: indices[i-1] + 1] + ['[/TRG]'] + sentence[indices[i-1] + 1: ]
            new_lbl += label[indices[prev_i]: indices[i-1] + 1] + ['O'] + label[indices[i-1] + 1: ]
        return new_sen, new_lbl
    def create_arg_data(self, jsn, global_count_dict, skip_history = True):
        turns_, turn_labels, turn_triggers, turn_events, turn_msrs = [], [], [], [], []
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            label_index = 0
            for event_type in jsn['events'][turns]:#each turn grouped by same event type
                if(event_type=='O' or event_type == "Amend_Action_Data"): 
                    continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                    new_sen, new_label = self.append_tags(sentence, trigger, label)
                    if(not skip_history):
                        turn_words = list(self.flatten([jsn["sentences"][i] + ['[SEP]'] for i in range(idx)] + new_sen if idx > 0 else new_sen))
                        turn_label = list(self.flatten([['O']*len(jsn["sentences"][i]) + ['O'] for i in range(idx)] + new_label if idx > 0 else new_label))
                    else:
                        turn_words = list(new_sen)
                        turn_label = list(new_label)
                    for x in turn_label:
                        if(self.labels_frequency.get(x) is None):
                            self.labels_frequency[x] = 0
                        self.labels_frequency[x] += 1
                    assert len(turn_words) == len(turn_label)
                    turns_.append(turn_words)
                    turn_labels.append(turn_label)
                    turn_triggers.append(trigger)
                    turn_events.append(event_type)
                    turn_msrs.append(metaSR)
        return {"all_sentences": turns_, "all_labels": turn_labels, "all_triggers": turn_triggers, "all_event_names": turn_events, "all_msrs": turn_msrs}
    def __init__(self, data_addr, tokenizer, max_len = 512, labels_to_ids=None):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.global_count_dict = {}
        self.labels_frequency = {}
        self.flatten = lambda *n: (e for a in n for e in (self.flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            thread_dict = self.create_arg_data(jsn, self.global_count_dict)
            turns, turn_labels, turn_triggers, turn_events, turn_msrs = thread_dict['all_sentences'], thread_dict['all_labels'], thread_dict['all_triggers'], thread_dict['all_event_names'], thread_dict['all_msrs']
            self.samples.extend([{'text': text,'arg_labels':label,"meta_semantic_role":msr, 'trigger_spans': ts, "event_name":en ,'file_name':os.path.split(file)[-1]} for text, label, msr, ts, en in zip(turns, turn_labels, turn_msrs, turn_triggers, turn_events)])
        ###
        self.samples = self.samples 
        self.len = len(self.samples)
        # self.__len__= len(self.samples)
        # print(f'------------------------------------------------LENGTH: {self.len}')
        if(labels_to_ids is not None):
            self.labels_to_ids = labels_to_ids
            self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        else:
            self.labels_to_ids = {}
            for label_id in self.labels_frequency.keys():
                if(self.labels_to_ids.get(label_id) is None):
                    self.labels_to_ids[label_id] = len(self.labels_to_ids)
        self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        ###       
    def get_labels_to_ids(self):
        return self.labels_to_ids, dict((v,k) for k, v in self.labels_to_ids.items())
    def get_labels_frequency_w_inverse_weights(self):
        labels_to_ids = self.labels_to_ids
        pos_weights = torch.tensor([0 for i in range(len(labels_to_ids))], dtype=torch.float)
        pos_weights_weighted = torch.tensor([0 for i in range(len(labels_to_ids))], dtype=torch.float)
        for label in labels_to_ids:
            l_freq = self.labels_frequency[label] if self.labels_frequency.get(label) is not None else 0
            pos_weights[labels_to_ids[label]] = 1/l_freq if l_freq != 0 else 0
        for class_, class_weight in self.labels_frequency.items():
            if(class_!=-100):
                total = sum([v for k, v in self.labels_frequency.items() if k!=-100])
                weight = math.ceil((total-class_weight)/class_weight)
                pos_weights_weighted[self.labels_to_ids[class_]] = weight
        return {"labels_frequency": self.labels_frequency, "inverse_pos_weights":pos_weights, 'weighted_pos': pos_weights_weighted}       
    def getSecondLastSEP(self, input_ids, labels):
        sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        sep_indices = [idx for idx, x in enumerate(input_ids) if x == sep_id]
        sep_indices = sep_indices[:len(sep_indices)-1]
        return sep_indices
    def nullifyHistory(self, input_ids, labels):
        sep_indices = self.getSecondLastSEP(input_ids, labels)
        if(len(sep_indices)==0):#the current email was very large so no history so no nullification
            return labels, sep_indices
        last_sep_index = sep_indices[-1]
        while(last_sep_index>0):
            labels[last_sep_index] = -100#torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)
            last_sep_index-=1
        return labels, sep_indices
    def get_loss_mask(self, sep_indices, current_label, pad_starts_from, len_event_ids):
        loss_mask = [1]*len(current_label)
        for idx in range(len_event_ids):
            loss_mask[idx] = 0#for [CLS]
        if(len(sep_indices)>0):
            last_sep_index = sep_indices[-1]
            while(last_sep_index>=0):
                loss_mask[last_sep_index] = 0
                last_sep_index -= 1
        pad_starts_from-=1#including [SEP]
        while(pad_starts_from<512):
            loss_mask[pad_starts_from] = 0
            pad_starts_from += 1
        return np.array(loss_mask)
    def get_meta_srs(self, extra_attribute, event_type):
        extra_attribute = extra_attribute.strip()
        if(event_type in ["Request_Meeting", "Request_Action"]):
            extra_attribute = "Other"
        else:
            if(event_type=="Request_Data" and extra_attribute == ""):
                extra_attribute = "Data Value"
            elif(event_type in ["Deliver_Data", "Deliver_Action_Data", "Deliver_Meeting_Data"] and extra_attribute==""):
                extra_attribute = "Positive"
            elif(event_type == "Request_Action_Data"):
                if(extra_attribute.find("Members")>=0):
                    extra_attribute = "Action Members"
                elif(extra_attribute.find("Time")>=0):
                    extra_attribute = "Action Time"
            elif(event_type.find("Amend")>=0 and extra_attribute==""):
                extra_attribute = "Update"
        #print(extra_attribute, '<<>>' ,event_type)
        assert extra_attribute.strip()!=""
        return extra_attribute
    def __getitem__(self, index):
        sentence = self.samples[index]["text"]
        file_name = self.samples[index]["file_name"]
        argument_labels = self.samples[index]["arg_labels"]
        event_trigger = self.samples[index]['trigger_spans']
        event_name = self.samples[index]['event_name']
        meta_semantic_role = self.samples[index]['meta_semantic_role']
        extra_type, extra_val = meta_semantic_role.split(' : ') if meta_semantic_role != "" else ("", "")
        #print(f"->{extra_type}<- ->{self.get_meta_srs(extra_val, event_name)}<-")
        extra_val = self.get_meta_srs(extra_val, event_name)
        if(self.labels_to_ids.get(extra_val) is None):
            #print(f"adding >{extra_val}< at index >{len(self.labels_to_ids)}<")
            self.ids_to_labels[len(self.labels_to_ids)] = extra_val
            self.labels_to_ids[extra_val] = len(self.labels_to_ids) 
        encoding = self.tokenizer(sentence, is_split_into_words = True, return_offsets_mapping = True)
        encoded_labels = list([-100]*len(encoding["offset_mapping"]))
        labels_ = []
        for lbl in argument_labels:
            if(self.labels_to_ids.get(lbl) is None):
                print(lbl, f" adding {lbl} at index {len(self.labels_to_ids)}")
                self.ids_to_labels[len(self.labels_to_ids)] = lbl
                self.labels_to_ids[lbl] = len(self.labels_to_ids)  
            labels_.append(self.labels_to_ids[lbl])
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if(mapping[0]!=0 and mapping[1]!=0):
                last_label = self.labels_to_ids[self.ids_to_labels[labels_[i-1]]] if self.ids_to_labels.get(labels_[i-1]) is not None else -100
                encoded_labels[idx] = last_label  
            elif(mapping[0]==0 and mapping[1]!=0):
                encoded_labels[idx]=labels_[i]
                i+=1
        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + encoding['input_ids'][-511:] if len(encoding['input_ids'])>512 else encoding['input_ids']
        labels = [-100] + encoded_labels[-511:] if len(encoded_labels)>512 else encoded_labels#cls has -100 so -511
        attention_mask = encoding['attention_mask'][-512:]
        ##
        #
        event_ids = self.tokenizer(event_name.replace("_", " ") + " [TYPE]")['input_ids'][:-1]
        len_event_ids = len(event_ids)
        input_ids = event_ids + input_ids[-512+len_event_ids:][1:] #+ ([tokenizer.convert_tokens_to_ids(tokenizer.sep_token)] if input_ids[-2] != tokenizer.convert_tokens_to_ids(tokenizer.pad_token) else [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)])
        labels = [-100]*len(event_ids) + labels[-512+len_event_ids:][1:]
        attention_mask = [1]*len(event_ids) + attention_mask[-512+len_event_ids:][1:]
        offset_mapping = [(-1, -1)]*len(event_ids) + encoding["offset_mapping"][-512+len_event_ids:][1:]
        #
        ##
        attention_mask += [0]*(512 - len(attention_mask))
        assert len(input_ids) == len(labels)
        input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(512 - len(input_ids))
        pad_starts_from = len(labels)
        labels += [-100]*(512 - len(labels))#fail safe for padding?
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'], sep_indices = self.nullifyHistory(input_ids, labels)
        loss_mask = self.get_loss_mask(sep_indices, item['labels'], pad_starts_from, len_event_ids)
        # ##
        # #
        # loss_mask = [0]*len(event_ids) + loss_mask[-512+len_event_ids:][1:]
        # #
        # ##
        for idx, input_id in enumerate(input_ids):
            if(input_id in [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token), self.tokenizer.convert_tokens_to_ids("[TRG]"), self.tokenizer.convert_tokens_to_ids("[/TRG]")]):
                item['labels'][idx] = -100#torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)  
            if(input_id in [self.tokenizer.convert_tokens_to_ids("[TRG]"), self.tokenizer.convert_tokens_to_ids("[/TRG]")]):
                loss_mask[idx] = 0
        #
        item['labels'] = torch.tensor(item['labels'], dtype=torch.long)
        item['input_ids'] = torch.tensor(input_ids)
        item['attention_mask'] = torch.tensor(attention_mask)
        item['token_type_ids'] = [0]*512
        item['loss_mask'] = torch.tensor(loss_mask)
        item["offset_mapping"] = torch.tensor(offset_mapping[-512:] + [(-1, -1)]*(512 - len(offset_mapping)))
        item['trigger_span'] = torch.tensor(self.tokenizer(event_trigger['words'], pad_to_max_length=True, max_length=512)['input_ids'])
        all_lengths = [len(item[x]) for x in item]
        try:
                len(set(all_lengths))==1
        except:
                print(item)
        item["msr_labels"] = torch.tensor(self.labels_to_ids[extra_val])
        item["file_name"] = torch.tensor(self.tokenizer(file_name, pad_to_max_length=True, max_length = self.max_len, truncation=True)['input_ids'])
        #print(item["meta_srs"])
        #print(all_lengths)
        # print("Event_Trigger: ", event_trigger)
        # print("Event_Name: ", event_name)
        # print("*"*50)
        # for idx, (x, y) in enumerate(zip(item['input_ids'], item['labels'])):
        #     print((tokenizer.convert_ids_to_tokens((x.tolist())) + '\t' + "LBL: " + str(ids_to_labels.get(y.tolist(), -100)) + '\t' + "AM: " + str(item["attention_mask"][idx]) + '\t' + "LM: " + str(item["loss_mask"][idx])).expandtabs(50))
        # print('*'*100)
        del item["token_type_ids"]
        return item
    def __len__(self):
        return self.len







class seq_tagger_arg_multi_class_test_data(Dataset):
    def append_tags(self, sentence, trigger):
        indices = trigger['indices']
        new_sen = sentence
        try:
            indices.remove("")
        except:
            pass
        if(len(indices)>0):
            #indices = [int(idx) for idx in indices]
            i = 1
            prev_i = 0
            new_sen = sentence[:indices[0]] + ['[TRG]']
            while(i<len(indices)):
                if(not indices[i]-1 == indices[i-1]):
                    new_sen += sentence[indices[prev_i]: indices[i-1] + 1] + ['[/TRG]'] + sentence[indices[i-1] + 1: indices[i]] + ['[TRG]']
                    prev_i = i
                i += 1
            new_sen += sentence[indices[prev_i]: indices[i-1] + 1] + ['[/TRG]'] + sentence[indices[i-1] + 1: ]
        return new_sen
    def find_context_email(self, email, history):
        length_email = len(email)
        first_email_word = email[0]
        for idx, history_word in enumerate(history):
            if(history_word==first_email_word):
                if(history[idx: idx+length_email]==email):
                    return range(idx, idx+length_email)

    def create_arg_test_data(self, jsn):
        #print(jsn)
        email_body = jsn["email_body"]
        predicted_triggers = jsn["predicted_triggers"]
        full_trigger_input = jsn["full_input"]
        for predicted_trigger in predicted_triggers:
            trigger_span = predicted_trigger["span"]
            trigger_index = predicted_trigger["indices"]
            event_type = predicted_trigger["type"]
            email_with_predicted_trigger = self.append_tags(email_body, predicted_trigger)#email_body[:trigger_index[0]] + ['[TRG]'] + email_body[trigger_index[0]: trigger_index[-1]+1] + ['[/TRG]'] + email_body[trigger_index[-1]+1:]
            history_index = self.find_context_email(email_body, full_trigger_input)
            history = full_trigger_input[:history_index[0]]#this gives the previous things
            new_input = history + email_with_predicted_trigger
            #print(new_input)
            self.samples.append({"input_text":  new_input, "event_type": event_type})
    def __init__(self, data_addr, tokenizer, max_len = 512, labels_to_ids=None):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_frequency = {}
        self.flatten = lambda *n: (e for a in n for e in (self.flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
        jsn = json.load(open(data_addr))
        #print(jsn)
        for predicted_trigger_dict in jsn:
            thread_dict = self.create_arg_test_data(predicted_trigger_dict)
            #print(thread_dict)
        ###
        self.len = len(self.samples)    
        self.labels_to_ids = labels_to_ids
        self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        ###       
    def get_labels_to_ids(self):
        return self.labels_to_ids, dict((v,k) for k, v in self.labels_to_ids.items())
    def __getitem__(self, index):
        sentence = self.samples[index]["input_text"]
        event_name = self.samples[index]['event_type']
        encoding = self.tokenizer.convert_tokens_to_ids(sentence)

        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + encoding[-511:] if len(encoding)>512 else encoding
        attention_mask = ([1]*len(encoding))[-512:]
        ##
        #
        event_ids = self.tokenizer(event_name.replace("_", " ") + " [TYPE]")['input_ids'][:-1]
        len_event_ids = len(event_ids)
        input_ids = event_ids + input_ids[-512+len_event_ids:][1:] #+ ([tokenizer.convert_tokens_to_ids(tokenizer.sep_token)] if input_ids[-2] != tokenizer.convert_tokens_to_ids(tokenizer.pad_token) else [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)])
        attention_mask = [1]*len(event_ids) + attention_mask[-512+len_event_ids:][1:]
        attention_mask += [0]*(512 - len(attention_mask))
        input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(512 - len(input_ids))
        item = {}
        item['input_ids'] = torch.tensor(input_ids)
        item['attention_mask'] = torch.tensor(attention_mask)
        return item
    def __len__(self):
        return self.len