from torch.utils.data import Dataset, DataLoader
from glob import glob
import os, json, copy
import torch
from functools import reduce
import numpy as np
from tqdm import tqdm
#from data_utils import *
from itertools import chain
import itertools, re, ast, math
class seq_tagger_trigger_once_dataset(Dataset):
    def extract_event_type(self, label):
        return label.replace("B-", '').replace("I-", '').strip()
    def get_multi_label_one_hot(self, label, labels_to_ids):
        return_label = []
        for lbl in label:
            #print(label)
            word_label = np.ones(len(labels_to_ids))*-100
            for l in lbl:
                if l!=-100:
                    word_label[l] = 1
            #if lbl!=-100:
            #    word_label[int(lbl.tolist()[0])] = 1
            return_label.append(torch.tensor(np.array([x if x!=-100 else 0 for x in word_label]), dtype = torch.float))
        #print(return_label)
        return torch.stack(return_label)
    def merge_labels_at_same_indices(self, turn_same_event_labels, event_name):
        tl = turn_same_event_labels[event_name]['trigger_label']
        if len(tl) > 1:
            multi_label = list(zip(*tl))
            for idx, label in enumerate(multi_label):
                uniques = set(label)
                #print('>', uniques,'<', label)
                if(len(uniques)>1):uniques.discard('O')
                #uniques = list(uniques.discard('O') if len(uniques)>1 else uniques)
                multi_label[idx] = list(uniques)
        else:
            multi_label = [[x] for x in tl[0]]
        return multi_label
    def clearTurnLabels(self, turn_labels):
        ret_labels = []
        for x in turn_labels:#a multu-label tuple
            if(len(x)>0 and type(x[0])==type(1)):
                x = [[xx] for xx in x]
            #print(x)
            x = list(set(list(itertools.chain.from_iterable([xx for xx in x]))))
            if(len(x)>1):
                try:
                    x.remove(-100)
                    x.remove('O')
                except Exception as e:
                    try:
                        x.remove('O')
                    except:
                        pass
                    pass
            ret_labels.append(x)
        return ret_labels
    def create_enum_data(self, jsn, global_count_dict, do_upsampling = False):
        turns_once, turn_labels_once = [], []
        input_ids, input_labels = [], []
        upsample = False
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            turn_dict, turn_same_event_labels = {}, {}
            turns_, turn_labels = [], []
            #ret_dict = {}
            for event_type in jsn['events'][turns]:#each turn grouped by same event type
                if(event_type=='O'): continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                #sort them here to assign in ascending order of indices
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                #triggers.sort(key = lambda x: int(x['indices'].split()[0]) if x['indices'].strip()!='' else float('inf'))#sort :D:D
                #print(triggers)
                if(event_type!='O'):
                    for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                        trigger_vector = ['O'] * len(sentence)
                        for idx_idx, trigger_index in enumerate(trigger['indices'].split()):
                            if(turn_dict.get(event_type) is None):
                                turn_dict[event_type] = {}
                            if(turn_dict[event_type].get(repeat_index) is None):
                                turn_dict[event_type][repeat_index] = len(turn_dict[event_type])
                            label_index = turn_dict[event_type][repeat_index]
                            if(global_count_dict.get(event_type) is None):
                                global_count_dict[event_type] = 0
                            if(global_count_dict[event_type]<label_index+1):
                                global_count_dict[event_type] = label_index+1
                            try:
                                trigger_index = int(trigger_index)
                                trigger_vector[trigger_index] = ('I-' if(idx_idx==0) else 'I-') + event_type + "_" + str(label_index)
                                #trigger_vector[trigger_index] = ('B-' if(idx_idx==0 or (trigger_index>0 and trigger_vector[trigger_index-1]=='O')) else 'I-') + event_type + "_" + str(label_index)
                            except:
                                pass
                        if(turn_same_event_labels.get(event_type) is None):
                            turn_same_event_labels[event_type] = {'sen': sentence, 'trigger_label':[]}
                        #turn_same_event_labels[event_type].append({'sen': sentence, 'trigger_label':[]})
                        turn_same_event_labels[event_type]['trigger_label'].append(trigger_vector)
                        #{'Deliver_Data': 18, 'Deliver_Action_Data': 11, 'Request_Action': 8, 'Request_Data': 5, 'Request_Meeting': 5, 'Request_Meeting_Data': 2, 'Deliver_Meeting_Data': 5, 'Amend_Data': 7, 'Request_Action_Data': 2, 'Amend_Meeting_Data': 1}
                        if(event_type in ["Request_Meeting", "Deliver_Meeting_Data", "Amend_Data", "Request_Data"]):
                            #print("setting true")
                            upsample = True
                        #verify =[(x, y) for x, y in zip(trigger_vector, sentence)]
                        #print(verify) 
                else: continue
                turn_label = self.merge_labels_at_same_indices(turn_same_event_labels, event_type)
                input_ids = list(chain(*[jsn['sentences'][i] + ['[SEP]'] for i in range(idx)])) + sentence if idx > 0 else sentence
                input_labels = list(chain(*[[-100]*(len(jsn['sentences'][i])+1) for i in range(idx)])) + turn_label if idx > 0 else turn_label
                assert len(input_ids) == len(input_labels)
                # if(ret_dict.get(event_type) is None):
                #     ret_dict[event_type] = []
                # data = {'text': input_ids, 'trigger_labels': input_labels, 'event_type': event_type}
                # ret_dict[event_type].append(data)
                # print(upsample)
                # if(upsample):
                #     for _ in range(3):
                #         turns_.append(input_ids)
                #         turn_labels.append({'labels': input_labels, 'event_type': event_type})
                turns_.append(input_ids)
                turn_labels.append({'labels': input_labels, 'event_type': event_type})
            all_labels = list(zip(*[turn_labels[i]['labels'] for i in range(len(turn_labels))]))
            #print(all_labels, '\n\n')
            new_labels = self.clearTurnLabels(all_labels)
            if(len(turns_)<=0):
                continue
            if(upsample and do_upsampling):
                for _ in range(2):
                    turns_once.append(input_ids)
                    turn_labels_once.append(new_labels)
            turns_once.append(input_ids)
            turn_labels_once.append(new_labels)
            # for x, y in zip(input_ids, new_labels):
            #    print(x, y)
            #    print('--------TURN END--------')
        return {'turns':turns_once, 'turn_labels':turn_labels_once}
    def __init__(self, data_addr, tokenizer, labels_to_ids, max_len = 512, do_upsampling = False):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.labels_to_ids = labels_to_ids
        self.ids_to_labels = dict((v,k) for k, v in labels_to_ids.items())
        self.max_len = max_len
        self.len = 0
        self.global_count_dict = {}
        self.do_upsampling = do_upsampling
        #read all the data files
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            #turns, turn_labels = self.extract_labels_and_turns(jsn)
            thread_dict = self.create_enum_data(jsn, self.global_count_dict, self.do_upsampling)
            turns, turn_labels = thread_dict['turns'], thread_dict['turn_labels']
            if(len(turns)<=0): continue
            # print(turns)
            # print('-------')
            # print(turn_labels)
            self.samples.extend([{'text': text,'trigger_labels':label, 'file_name':os.path.split(file)[-1]} for text, label in zip(turns, turn_labels)])
        self.samples = self.samples 
        self.len = len(self.samples)
        print('Data Loaded. Distribution is as follows: ')
        print(self.global_count_dict) 
    def getSecondLastSEP(self, input_ids, labels):
        sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        sep_indices = [idx for idx, x in enumerate(input_ids) if x == sep_id]
        sep_indices = sep_indices[:len(sep_indices)-1]
        #print(sep_indices)
        return sep_indices
    def nullifyHistory(self, input_ids, labels):
        #we need to start from second last SEP
        sep_indices = self.getSecondLastSEP(input_ids, labels)
        #print(sep_indices)
        if(len(sep_indices)==0):#the current email was very large so no history so no nullification
            return labels, sep_indices
        last_sep_index = sep_indices[-1]
        while(last_sep_index>0):
            labels[last_sep_index] = torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)
            last_sep_index-=1
        return labels, sep_indices
    def get_loss_mask(self, sep_indices, current_label, pad_starts_from):
        #print(sep_indices)
        loss_mask = [1]*len(current_label)
        loss_mask[0] = 0#for [CLS]
        if(len(sep_indices)>0):
            last_sep_index = sep_indices[-1]
            while(last_sep_index>=0):
                loss_mask[last_sep_index] = 0
                last_sep_index -= 1
        pad_starts_from-=1#including [SEP]
        while(pad_starts_from<self.max_len):
            loss_mask[pad_starts_from] = 0
            pad_starts_from += 1
        return np.array(loss_mask)
    def __getitem__(self, index):
        sample = self.samples[index]
        complete_email, complete_trigger_labels, file_name = sample['text'], sample['trigger_labels'], sample['file_name']#, sample['event_type']
        #print(complete_trigger_labels)
        complete_email = [word if (word!=[''] and word!='')  else '*' for word in complete_email]
        complete_trigger_labels = [[x] if type(x)==int else x for x in complete_trigger_labels]
        encoding = self.tokenizer(complete_email, is_split_into_words = True, return_offsets_mapping = True)
        #encoded_labels = list(np.ones(len(encoding["offset_mapping"]), dtype = int)*-100)
        encoded_labels = list([[-100]]*len(encoding["offset_mapping"]))
        #labels_ = [self.labels_to_ids[label] if self.labels_to_ids.get(label) is not None else -100 for label in complete_trigger_labels]
        labels_ = []
        #print(complete_trigger_labels)
        for lbl in complete_trigger_labels:
            word_labels = []
            for l in lbl:
                    enc_lbl = self.labels_to_ids[l] if self.labels_to_ids.get(l) is not None else -100
                    word_labels.append(enc_lbl)
            labels_.append(word_labels)
        #print(labels_)
        i = 0
        #print(encoding)
        #print(self.labels_to_ids)
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if(mapping[0]!=0 and mapping[1]!=0):
                last_label = [self.labels_to_ids[self.ids_to_labels[x].replace('B-', 'I-')]  if self.ids_to_labels.get(x) is not None else -100 for x in labels_[i-1]]
                encoded_labels[idx] = last_label #assign I- of the "last_event" labels to sub-words 
            elif(mapping[0]==0 and mapping[1]!=0):
                encoded_labels[idx]=labels_[i]
                # try:    
                #     encoded_labels[idx]=labels_[i]
                # except:
                #     print(encoded_labels)
                #     print(labels_)
                #     print(complete_trigger_labels)
                #     print(complete_email)
                #     print(file_name)
                i+=1
            #print(idx, mapping, self.tokenizer.decode(encoding['input_ids'][idx]), encoded_labels[idx], labels_[i])
        #print(encoded_labels)
        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + encoding['input_ids'][-511:] if len(encoding['input_ids'])>512 else encoding['input_ids']
        labels = [[-100]] + encoded_labels[-511:] if len(encoded_labels)>512 else encoded_labels#cls has -100 so -511
        #print(labels)
        attention_mask = encoding['attention_mask'][-512:]
        attention_mask += [0]*(self.max_len - len(attention_mask))
        assert len(input_ids) == len(labels)
        input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(self.max_len - len(input_ids))
        pad_starts_from = len(labels)
        labels += [[-100]]*(self.max_len - len(labels))#fail safe for padding?
        # if(index==5):
        #     print(labels)
        #     print(labels_)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        #print(labels)
        #print(complete_trigger_labels)
        item['labels'], sep_indices = self.nullifyHistory(input_ids, self.get_multi_label_one_hot(labels, self.labels_to_ids))
        loss_mask = self.get_loss_mask(sep_indices, item['labels'], pad_starts_from)
        for idx, input_id in enumerate(input_ids):
            if(input_id in [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)]):
                item['labels'][idx] = torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)
        
        # event_ids = self.tokenizer(f'{event_type} [TYPE]'.split('_'), is_split_into_words = True)['input_ids'][:-1]
        # input_ids[:len(event_ids)] = event_ids
        # loss_mask[:len(event_ids)] = [0]*len(event_ids)
        item['input_ids'] = torch.as_tensor(input_ids)
        #item["split_sen"] = self.tokenizer.convert_ids_to_tokens(input_ids)
        #item["file_name"] = file_name
        #item["original_text"] = {'original_text': complete_email, 'len_of_original_text': len(complete_email)}
        item['attention_mask'] = torch.tensor(attention_mask)
        item['token_type_ids'] = [0]*self.max_len
        item['loss_mask'] = torch.tensor(loss_mask)
        #print(item.keys(), '<<<<<<<<<<<<<<<<<<<<<<<<TO_DELETE')
        del item['offset_mapping']
        del item['token_type_ids']
        #del item['offset_mapping']
        return item
    def __len__(self):
        return self.len


class seq_tagger_trigger_once_dataset_id_only(Dataset):
    def extract_event_type(self, label):
        return label.replace("B-", '').replace("I-", '').strip()
    def get_multi_label_one_hot(self, label, labels_to_ids):
        return_label = []
        for lbl in label:
            word_label = np.ones(len(labels_to_ids))*-100
            for l in lbl:
                if l!=-100:
                    word_label[l] = 1
            return_label.append(torch.tensor(np.array([x if x!=-100 else 0 for x in word_label]), dtype = torch.float))
        return torch.stack(return_label)
    def merge_labels_at_same_indices(self, turn_same_event_labels, event_name):
        tl = turn_same_event_labels[event_name]['trigger_label']
        if len(tl) > 1:
            multi_label = list(zip(*tl))
            for idx, label in enumerate(multi_label):
                uniques = set(label)
                if(len(uniques)>1):uniques.discard('O')
                multi_label[idx] = list(uniques)
        else:
            multi_label = [[x] for x in tl[0]]
        return multi_label
    def clearTurnLabels(self, turn_labels):
        ret_labels = []
        for x in turn_labels:#a multu-label tuple
            if(len(x)>0 and type(x[0])==type(1)):
                x = [[xx] for xx in x]
            x = list(set(list(itertools.chain.from_iterable([xx for xx in x]))))
            if(len(x)>1):
                try:
                    x.remove(-100)
                    x.remove('O')
                except Exception as e:
                    try:
                        x.remove('O')
                    except:
                        pass
                    pass
            ret_labels.append(x)
        return ret_labels
    def create_enum_data(self, jsn, global_count_dict, do_upsampling = False):
        turns_once, turn_labels_once, all_triggers = [], [], []
        input_ids, input_labels = [], []
        upsample = False
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            label_index = 0
            turn_dict, turn_same_event_labels = {}, {}
            turns_, turn_labels = [], []
            for event_type in jsn['events'][turns]:#each turn grouped by same event type
                if(event_type=='O'): continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                #sort them here to assign in ascending order of indices
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                triggers.sort(key = lambda x: int(x['indices'].split()[0]) if x['indices'].strip()!='' else float('inf'))#sort :D:D
                all_triggers.append(triggers)
                if(event_type!='O'):
                    event_type = 'I'
                    for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                        trigger_vector = ['O'] * len(sentence)
                        for idx_idx, trigger_index in enumerate(trigger['indices'].split()):
                            # if(turn_dict.get(event_type) is None):
                            #     turn_dict[event_type] = {}
                            # if(turn_dict[event_type].get(repeat_index) is None):
                            #     turn_dict[event_type][repeat_index] = len(turn_dict[event_type])
                            # label_index = turn_dict[event_type][repeat_index]
                            if(global_count_dict.get(event_type) is None):
                                global_count_dict[event_type] = 0
                            if(global_count_dict[event_type]<label_index+1):
                                global_count_dict[event_type] = label_index+1
                            try:
                                trigger_index = int(trigger_index)
                                label_ll = event_type + "_" + str(label_index)
                                trigger_vector[trigger_index] =  label_ll
                                if(self.labels_frequency.get(label_ll) is None):
                                    self.labels_frequency[label_ll] = 0
                                self.labels_frequency[label_ll] += 1
                            except:
                                pass
                        label_index += 1
                        if(turn_same_event_labels.get(event_type) is None):
                            turn_same_event_labels[event_type] = {'sen': sentence, 'trigger_label':[]}
                        turn_same_event_labels[event_type]['trigger_label'].append(trigger_vector)
                        if(event_type in ["Request_Meeting", "Deliver_Meeting_Data", "Amend_Data", "Request_Data"]):
                            upsample = True
                else: continue
                turn_label = self.merge_labels_at_same_indices(turn_same_event_labels, event_type)
                input_ids = list(chain(*[jsn['sentences'][i] + ['[SEP]'] for i in range(idx)])) + sentence if idx > 0 else sentence
                input_labels = list(chain(*[[-100]*(len(jsn['sentences'][i])+1) for i in range(idx)])) + turn_label if idx > 0 else turn_label
                assert len(input_ids) == len(input_labels)
                turns_.append(input_ids)
                turn_labels.append({'labels': input_labels, 'event_type': event_type})
            all_labels = list(zip(*[turn_labels[i]['labels'] for i in range(len(turn_labels))]))
            new_labels = self.clearTurnLabels(all_labels)
            if(len(turns_)<=0):
                continue
            if(upsample and do_upsampling):
                for _ in range(2):
                    turns_once.append(input_ids)
                    turn_labels_once.append(new_labels)
            turns_once.append(input_ids)
            turn_labels_once.append(new_labels)
        return {'turns':turns_once, 'turn_labels':turn_labels_once, 'all_triggers': all_triggers}
    def __init__(self, data_addr, tokenizer, max_len = 512, do_upsampling = False, labels_to_ids=None):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = 0
        self.global_count_dict = {}
        self.do_upsampling = do_upsampling
        self.labels_frequency = {}
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            thread_dict = self.create_enum_data(jsn, self.global_count_dict, self.do_upsampling)
            turns, turn_labels = thread_dict['turns'], thread_dict['turn_labels']
            if(len(turns)<=0): continue
            self.samples.extend([{'text': text,'trigger_labels':label, 'trigger_words': tw,'file_name':os.path.split(file)[-1]} for text, label, tw in zip(turns, turn_labels, thread_dict['all_triggers'])])
        self.samples = self.samples 
        self.len = len(self.samples)
        print('Data Loaded. Distribution is as follows: ')
        print(self.global_count_dict) 
        if(labels_to_ids is not None):
            self.labels_to_ids = labels_to_ids
        else:
            self.labels_to_ids = {}
            for label in self.global_count_dict:
                label_counts = self.global_count_dict[label]
                for i in range(label_counts):
                    #labels_to_ids['B-' + label + "_" + str(i)] = len(labels_to_ids)
                    self.labels_to_ids[label + "_" + str(i)] = len(self.labels_to_ids)
        self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        #print(">>>>>>>>>>", self.labels_frequency)
    def get_labels_to_ids(self):
        return self.labels_to_ids, self.ids_to_labels
    def get_labels_frequency_w_inverse_weights(self):
        labels_to_ids = self.labels_to_ids
        pos_weights = torch.tensor([0 for i in range(len(labels_to_ids))], dtype=torch.float)
        for label in labels_to_ids:
            l_freq = self.labels_frequency[label]
            pos_weights[labels_to_ids[label]] = 1/l_freq
        return self.labels_frequency, pos_weights       
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
            labels[last_sep_index] = torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)
            last_sep_index-=1
        return labels, sep_indices
    def get_loss_mask(self, sep_indices, current_label, pad_starts_from):
        loss_mask = [1]*len(current_label)
        loss_mask[0] = 0#for [CLS]
        if(len(sep_indices)>0):
            last_sep_index = sep_indices[-1]
            while(last_sep_index>=0):
                loss_mask[last_sep_index] = 0
                last_sep_index -= 1
        pad_starts_from-=1#including [SEP]
        while(pad_starts_from<self.max_len):
            loss_mask[pad_starts_from] = 0
            pad_starts_from += 1
        return np.array(loss_mask)
    def order_labels(self, labels):
        indices_dict = {}
        for lab_pos, x in enumerate(labels):
            if(x==[-100] or x == ['O']):
                continue
            assert 'O' not in x and -100 not in x
            for xx in x:
                if indices_dict.get(xx) is None:
                    indices_dict[xx] = []
                indices_dict[xx].append(lab_pos)
        counter = 0
        for lbl in indices_dict:
            indices_in_labels = indices_dict[lbl]
            for index in indices_in_labels:
                assert lbl in labels[index]
                labels[index].remove(lbl)
                labels[index].append('I_' + str(counter))
            counter+=1
        return labels
    def __getitem__(self, index):
        sample = self.samples[index]
        complete_email, complete_trigger_labels, file_name = sample['text'], sample['trigger_labels'], sample['file_name']#, sample['event_type']
        complete_email = [word if (word!=[''] and word!='')  else '*' for word in complete_email]
        complete_trigger_labels = [[x] if type(x)==int else x for x in complete_trigger_labels]
        #print('>>>>>>>>>>', sentence, '<<<<<<<<<<<')
        #print(sample['trigger_words'])
        # print('Before', complete_trigger_labels)
        complete_trigger_labels = self.order_labels(complete_trigger_labels)
        # print('After',complete_trigger_labels)
        # if(sample['file_name']=='blair-l_inbox87.json'):
        #     print('after order', complete_trigger_labels)
        #     print(complete_email)
        encoding = self.tokenizer(complete_email, is_split_into_words = True, return_offsets_mapping = True)
        encoded_labels = list([[-100]]*len(encoding["offset_mapping"]))
        labels_ = []
        for lbl in complete_trigger_labels:
            word_labels = []
            for l in lbl:
                    enc_lbl = self.labels_to_ids[l] if self.labels_to_ids.get(l) is not None else -100
                    word_labels.append(enc_lbl)
            labels_.append(word_labels)
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if(mapping[0]!=0 and mapping[1]!=0):
                last_label = [self.labels_to_ids[self.ids_to_labels[x]]  if self.ids_to_labels.get(x) is not None else -100 for x in labels_[i-1]]
                encoded_labels[idx] = last_label #assign I- of the "last_event" labels to sub-words 
            elif(mapping[0]==0 and mapping[1]!=0):
                encoded_labels[idx]=labels_[i]
                i+=1
        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + encoding['input_ids'][-511:] if len(encoding['input_ids'])>512 else encoding['input_ids']
        labels = [[-100]] + encoded_labels[-511:] if len(encoded_labels)>512 else encoded_labels#cls has -100 so -511
        attention_mask = encoding['attention_mask'][-512:]
        attention_mask += [0]*(self.max_len - len(attention_mask))
        assert len(input_ids) == len(labels)
        input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(self.max_len - len(input_ids))
        pad_starts_from = len(labels)
        labels += [[-100]]*(self.max_len - len(labels))#fail safe for padding?
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'], sep_indices = self.nullifyHistory(input_ids, self.get_multi_label_one_hot(labels, self.labels_to_ids))
        loss_mask = self.get_loss_mask(sep_indices, item['labels'], pad_starts_from)
        for idx, input_id in enumerate(input_ids):
            if(input_id in [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)]):
                item['labels'][idx] = torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)
        
        item['input_ids'] = torch.as_tensor(input_ids)
        item['attention_mask'] = torch.tensor(attention_mask)
        item['token_type_ids'] = [0]*self.max_len
        item['loss_mask'] = torch.tensor(loss_mask)
        del item["token_type_ids"]
        del item['offset_mapping']
        return item
    def __len__(self):
        return self.len

class seq_tagger_trigger_once_dataset_id_class(Dataset):
    def append_tags(self, sentence, trigger):
        indices = trigger['indices'].split(" ")
        new_sen = sentence
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
            while(i<len(indices)):
                if(not indices[i]-1 == indices[i-1]):
                    new_sen += sentence[indices[prev_i]: indices[i-1] + 1] + ['[/TRG]'] + sentence[indices[i-1] + 1: indices[i]] + ['[TRG]']
                    prev_i = i
                i += 1
            new_sen += sentence[indices[prev_i]: indices[i-1] + 1] + ['[/TRG]'] + sentence[indices[i-1] + 1: ]
        return new_sen
    def create_class_data(self, jsn, global_count_dict, do_upsampling = False):
        turns_once, turn_labels_once = [], []
        input_ids, input_labels = [], []
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            turn_dict, turn_same_event_labels = {}, {}
            turns_, turn_labels = [], []
            for event_type in jsn['events'][turns]:#each turn grouped by same event type
                if(event_type=='O' or event_type=="Amend_Action_Data"): continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                #sort them here to assign in ascending order of indices
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                triggers.sort(key = lambda x: int(x['indices'].split()[0]) if x['indices'].strip()!='' else float('inf'))#sort :D:D
                if(event_type!='O'):
                    if(self.labels_to_ids.get(event_type) is None):
                        self.labels_to_ids[event_type] = len(self.labels_to_ids)
                    if(self.labels_frequency.get(event_type) is None):
                        self.labels_frequency[event_type] = 0
                    self.labels_frequency[event_type] += 1
                    for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                        new_sen = self.append_tags(sentence, trigger)
                        #print(idx, new_sen, trigger)
                        new_turn = list(chain(*[jsn['sentences'][i] + ['[SEP]'] for i in range(idx)])) + new_sen if idx > 0 else new_sen
                        #new_turn = turns_once[-1] + ['SEP'] + [new_sen] if idx > 0 else [new_sen]
                        turns_once.append(new_turn)
                        turn_labels_once.append(event_type)
        return turns_once, turn_labels_once
    def get_labels_frequency_w_inverse_weights(self):
        labels_to_ids = self.labels_to_ids
        pos_weights = torch.tensor([0 for i in range(len(labels_to_ids))], dtype=torch.float)
        for label in labels_to_ids:
            l_freq = self.labels_frequency[label]
            pos_weights[labels_to_ids[label]] = 1/l_freq
        return self.labels_frequency, pos_weights
    def __init__(self, data_addr, tokenizer, max_len = 512, do_upsampling = False, labels_to_ids=None):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = 0
        self.global_count_dict = {}
        self.do_upsampling = do_upsampling
        self.labels_frequency = {}
        if(labels_to_ids is not None):
            self.labels_to_ids = labels_to_ids
        else:
            self.labels_to_ids = {}
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            turns, turn_labels = self.create_class_data(jsn, self.global_count_dict, self.do_upsampling)
            self.samples.extend([ [t, tl] for t, tl in zip(turns, turn_labels) ])
        #print(self.labels_to_ids)
    def __getitem__(self, index):
        sample = self.samples[index]
        complete_email, turn_label = sample[0], sample[1]#, sample['file_name']#, sample['event_type']
        complete_email = [word if (word!=[''] and word!='')  else '*' for word in complete_email]
        encoding = self.tokenizer(complete_email, is_split_into_words = True, padding='max_length', truncation=True)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        # if(self.labels_to_ids.get(turn_label) is None):
        #     self.labels_to_ids[turn_label] = len(self.labels_to_ids)
        # print(turn_label, self.labels_to_ids[turn_label])
        turn_label = self.labels_to_ids[turn_label]
        item['label'] = torch.tensor(turn_label)
        del_items = []
        item['loss_mask'] = []
        for key, value in item.items():
            if(key in ["input_ids", "label", 'loss_mask', "attention_mask"]): 
                continue
            del_items.append(key)
        for k in del_items:
            del item[k]
        return item
    def get_labels_to_ids(self):
        return self.labels_to_ids, dict([(value, key) for key, value in self.labels_to_ids.items()])
    def __len__(self):
        return len(self.samples)



class seq_tagger_trigger_multi_class_con_sep(Dataset):
    def extract_event_type(self, label):
        return label.replace("B-", '').replace("I-", '').strip()
    def merge_labels_at_same_indices(self, turn_same_event_labels, event_name):
        tl = turn_same_event_labels[event_name]['trigger_label']
        if len(tl) > 1:
            multi_label = list(zip(*tl))
            for idx, label in enumerate(multi_label):
                uniques = set(label)
                if(len(uniques)>1):uniques.discard('O')
                multi_label[idx] = list(uniques)
        else:
            multi_label = [[x] for x in tl[0]]
        return multi_label
    def clearTurnLabels(self, turn_labels):
        ret_labels = []
        for x in turn_labels:#a multu-label tuple
            if(len(x)>0 and type(x[0])==type(1)):
                x = [[xx] for xx in x]
            x = list(set(list(itertools.chain.from_iterable([xx for xx in x]))))
            if(len(x)>1):
                try:
                    x.remove(-100)
                    x.remove('O')
                except Exception as e:
                    try:
                        x.remove('O')
                    except:
                        pass
                    pass
            ret_labels.append(x)
        return ret_labels
    def create_multi_class(self, label, sentence):
        # print(label)
        ret_label = [x if type(x)==type(1) else x[0] if(len(x)==1) else re.sub('_[\d]+', '', ('S_' + x[0].replace('B_','').replace('I_', ''))) for x in label]
        ret_label = [re.sub('_[\d]+', '', x) if type(x) == type("xxx") else x for x in ret_label]
        # print('-'*100)
        # print(sentence)
        # print('_'*100)
        # for label in ret_label:
        #     if(type(label) == type("") and label.find("S_")>=0):
        #         print(ret_label)
        #         print(sentence)
        #         exit
        for x in ret_label:
            if(self.labels_frequency.get(x) is None):
                self.labels_frequency[x] = 0
            self.labels_frequency[x] += 1

        return ret_label
        return ret_label
    def create_enum_data(self, jsn, global_count_dict, do_upsampling = False, skip_history = False):
        turns_once, turn_labels_once, all_triggers = [], [], []
        input_ids, input_labels = [], []
        upsample = False
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            label_index = 0
            turn_dict, turn_same_event_labels = {}, {}
            turns_, turn_labels, turn_triggers = [], [], []
            for event_type in jsn['events'][turns]:#each turn grouped by same event type
                if(event_type=='O' or event_type == "Amend_Action_Data"): 
                    continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                #sort them here to assign in ascending order of indices
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                triggers.sort(key = lambda x: int(x['indices'].split()[0]) if x['indices'].strip()!='' else float('inf'))#sort :D:D
                self.all_triggers.extend(triggers)
                if(event_type!='O'):
                    # event_type = 'I'
                    # print(">>>>>>", sentence, triggers, len(turn_triggers), turn_triggers)
                    non_zero_counts = sum([1 for tt in triggers if tt['words'].strip()==""])
                    # print(event_type)
                    # if(non_zero_counts>0):
                    #     print('-'*100)
                    #     print(sum([1 for tt in triggers if tt['words'].strip()==""]))
                    #     print(triggers)
                    #     print(sentence)
                    #     print('-'*100)
                    for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                        trigger_vector = ['O'] * len(sentence)
                        turn_triggers.append(trigger['words'])
                        for idx_idx, trigger_index in enumerate(trigger['indices'].split()):
                            # if(global_count_dict.get(event_type) is None):
                            #     global_count_dict[event_type] = 0
                            # if(global_count_dict[event_type]<label_index+1):
                            #     global_count_dict[event_type] = label_index+1
                            try:
                                trigger_index = int(trigger_index)
                                label_ll = ('B' if idx_idx ==0 else 'I') + f"_{event_type}_" + str(label_index)#<<<----- option 2
                                trigger_vector[trigger_index] =  label_ll
                                # if(self.labels_frequency.get(label_ll) is None):
                                #     self.labels_frequency[label_ll] = 0
                                # self.labels_frequency[label_ll] += 1
                            except Exception as e:
                                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", e)
                                pass
                        label_index += 1
                        if(turn_same_event_labels.get(event_type) is None):
                            turn_same_event_labels[event_type] = {'sen': sentence, 'trigger_label':[]}
                        turn_same_event_labels[event_type]['trigger_label'].append(trigger_vector)
                        if(event_type in ["Request_Meeting", "Deliver_Meeting_Data", "Amend_Data", "Request_Data"]):
                            upsample = True
                else: continue
                turn_label = self.merge_labels_at_same_indices(turn_same_event_labels, event_type)
                #print(turn_label)
                if(not skip_history):
                    input_ids = list(chain(*[jsn['sentences'][i] + ['[CONTEXT]'] for i in range(idx)])) + ["[SEP]"] + sentence if idx > 0 else sentence
                    input_labels = list(chain(*[[-100]*(len(jsn['sentences'][i])+1) for i in range(idx)])) + [-100] + turn_label if idx > 0 else turn_label
                else:
                    input_ids = sentence
                    input_labels = turn_label
                assert len(input_ids) == len(input_labels)
                turns_.append(input_ids)
                turn_labels.append({'labels': input_labels, 'event_type': event_type})
            turn_triggers = [x for x in turn_triggers if (x!="" and x!=[""])]
            if(len(turn_triggers)>0):
                all_triggers.append(turn_triggers)
            #print('<><><><><>---->>>', sentence, turn_triggers)
            all_labels = list(zip(*[turn_labels[i]['labels'] for i in range(len(turn_labels))]))
            #print(all_labels)
            new_labels = self.clearTurnLabels(all_labels)
            #print(new_labels)
            #print('*'*100)
            new_labels = self.create_multi_class(new_labels, sentence)
            if(len(turns_)<=0):
                continue
            if(upsample and do_upsampling):
                for _ in range(2):
                    turns_once.append(input_ids)
                    turn_labels_once.append(new_labels)
            turns_once.append(input_ids)
            turn_labels_once.append(new_labels)
        #print(len(all_triggers))
        # print(turns_once, len(turns_once))
        # print(turn_labels_once, len(turn_labels_once))
        # print(all_triggers, len(all_triggers))
        return {'turns':turns_once, 'turn_labels':turn_labels_once, 'turn_triggers': all_triggers}
    def __init__(self, data_addr, tokenizer, max_len = 512, do_upsampling = False, labels_to_ids=None):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.all_special_tokens = list(tokenizer.special_tokens_map.values()) + list(tokenizer.get_added_vocab().keys())
        self.max_len = max_len
        self.len = 0
        self.global_count_dict = {}
        self.do_upsampling = do_upsampling
        self.labels_frequency = {}
        self.all_triggers = []
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            thread_dict = self.create_enum_data(jsn, self.global_count_dict, self.do_upsampling)
            turns, turn_labels = thread_dict['turns'], thread_dict['turn_labels']
            if(len(turns)<=0): continue
            self.samples.extend([{'text': text,'trigger_labels':label, 'trigger_spans': ts,'file_name':os.path.split(file)[-1]} for text, label, ts in zip(turns, turn_labels, thread_dict['turn_triggers'])])
        self.samples = self.samples 
        self.len = len(self.samples)
        # print('Data Loaded. Distribution is as follows: ')
        # print(self.global_count_dict) 
        with open("trigger_train_data.json", "w") as f:
            json.dump(self.samples, f, indent = 4)
        if(labels_to_ids is not None):
            self.labels_to_ids = labels_to_ids
        else:
            self.labels_to_ids = {'O':0}
            for label in self.labels_frequency:
                label_counts = self.labels_frequency[label]
                #for i in range(label_counts):
                if(label==-100 or self.labels_to_ids.get(label) is not None): continue
                self.labels_to_ids[label] = len(self.labels_to_ids)
        self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        # print(">>>>>>>", self.labels_frequency, "<<<<<<")
    def print_all_triggers(self):
        print(self.all_triggers)
    def get_labels_to_ids(self):
        return self.labels_to_ids, self.ids_to_labels
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
        sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
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
    def get_loss_mask(self, sep_indices, current_label, pad_starts_from):
        loss_mask = [1]*len(current_label)
        loss_mask[0] = 0#for [CLS]
        if(len(sep_indices)>0):
            last_sep_index = sep_indices[-1]
            while(last_sep_index>=0):
                loss_mask[last_sep_index] = 0
                last_sep_index -= 1
        pad_starts_from-=1#including [SEP]
        while(pad_starts_from<self.max_len):
            loss_mask[pad_starts_from] = 0
            pad_starts_from += 1
        return np.array(loss_mask)
    def __getitem__(self, index):
        sample = self.samples[index]
        complete_email, complete_trigger_labels, file_name, trigger_spans = sample['text'], sample['trigger_labels'], sample['file_name'], sample['trigger_spans']#, sample['event_type']
        complete_email = [word if (word!=[''] and word!='')  else '*' for word in complete_email]
        #complete_email = [word.lower() if word not in self.all_special_tokens else word for word in complete_email]
        # print([(w, l) for w, l in zip(complete_email, complete_trigger_labels)])
        # print('-'*100)
        #complete_trigger_labels = [[x] if type(x)==int else x for x in complete_trigger_labels]
        #complete_trigger_labels = self.order_labels(complete_trigger_labels)
        # print(trigger_spans)
        #print(complete_trigger_labels)
        # print(complete_email)
        # print('-'*100)
        encoding = self.tokenizer(complete_email, is_split_into_words = True, return_offsets_mapping = True)
        #print(self.tokenizer.decode(encoding['input_ids']))
        encoded_labels = list([-100]*len(encoding["offset_mapping"]))
        labels_ = []
        for lbl in complete_trigger_labels:
            #word_labels = []
            #for l in lbl:
            #        enc_lbl = self.labels_to_ids[l] if self.labels_to_ids.get(l) is not None else -100
            #        word_labels.append(enc_lbl)
            labels_.append(self.labels_to_ids.get(lbl))
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if(mapping[0]!=0 and mapping[1]!=0):
                last_label = self.labels_to_ids[self.ids_to_labels[labels_[i-1]]] if self.ids_to_labels.get(labels_[i-1]) is not None else -100 #[self.labels_to_ids[self.ids_to_labels[x]]  if self.ids_to_labels.get(x) is not None else -100 for x in labels_[i-1]]
                encoded_labels[idx] = last_label #assign I- of the "last_event" labels to sub-words 
            elif(mapping[0]==0 and mapping[1]!=0):
                encoded_labels[idx]=labels_[i]
                i+=1
        #print(encoded_labels)
        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + encoding['input_ids'][-511:] if len(encoding['input_ids'])>512 else encoding['input_ids']
        labels = [-100] + encoded_labels[-511:] if len(encoded_labels)>512 else encoded_labels#cls has -100 so -511
        attention_mask = encoding['attention_mask'][-512:]
        attention_mask += [0]*(self.max_len - len(attention_mask))
        assert len(input_ids) == len(labels)
        input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(self.max_len - len(input_ids))
        pad_starts_from = len(labels)
        labels += [-100]*(self.max_len - len(labels))#fail safe for padding?
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'], sep_indices = self.nullifyHistory(input_ids, labels)
        loss_mask = self.get_loss_mask(sep_indices, item['labels'], pad_starts_from)
        for idx, input_id in enumerate(input_ids):
            if(input_id in [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token), self.tokenizer.convert_tokens_to_ids("[CONTEXT]")]):
                item['labels'][idx] = -100#torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)        
        #print(item["labels"])
        item['labels'] = torch.tensor(item['labels'], dtype=torch.long)
        item['input_ids'] = torch.tensor(input_ids)
        item['attention_mask'] = torch.tensor(attention_mask)
        item['token_type_ids'] = [0]*self.max_len
        item['loss_mask'] = torch.tensor(loss_mask)
        item["offset_mapping"] = torch.tensor( item["offset_mapping"].tolist()[-512:] + [(-1, -1)]*(self.max_len - len(item["offset_mapping"])))
        #print(self.tokenizer.decode(self.tokenizer('[SEP]'.join(trigger_spans))['input_ids']))
        item['trigger_span'] = torch.tensor(self.tokenizer('[SEP]'.join(trigger_spans), pad_to_max_length=True, max_length=512)['input_ids'])
        item["file_name"] = torch.tensor(self.tokenizer(file_name, pad_to_max_length=True, max_length = self.max_len, truncation=True)['input_ids'])
        #x = torch.tensor([101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # print("Complete email: ", complete_email, '\nTS: ', trigger_spans)
        # print("TS: ", item['trigger_span'])
        # if(sum(item['trigger_span']) == 101+102):
        #     tokenized_text = self.tokenizer.convert_ids_to_tokens(item['input_ids'][item['loss_mask']==1])
        #     print("check this: ", trigger_spans)
        #     text_label_pair = [(tt, self.ids_to_labels[lbl] if lbl!=-100 else -100) for tt, lbl in zip(tokenized_text, item['labels'][item['loss_mask']==1].tolist())]
        #     print("TL: ", text_label_pair)
            #print("labels: ", item['labels'].tolist())
        #print(item["offset_mapping"].shape)
        del item["token_type_ids"]
        #del item['offset_mapping']
        return item
    def __len__(self):
        return self.len


# xx = [( ids_to_labels[x.tolist()], y, str(z.tolist()) ) for (x,y,z) in zip(labels,sentence, offset_mapping)]
# for x in xx:
#     print(x)



class seq_tagger_trigger_multi_class(Dataset):
    def extract_event_type(self, label):
        return label.replace("B-", '').replace("I-", '').strip()
    def merge_labels_at_same_indices(self, turn_same_event_labels, event_name):
        tl = turn_same_event_labels[event_name]['trigger_label']
        if len(tl) > 1:
            multi_label = list(zip(*tl))
            for idx, label in enumerate(multi_label):
                uniques = set(label)
                if(len(uniques)>1):uniques.discard('O')
                multi_label[idx] = list(uniques)
        else:
            multi_label = [[x] for x in tl[0]]
        return multi_label
    def clearTurnLabels(self, turn_labels):
        ret_labels = []
        for x in turn_labels:#a multu-label tuple
            if(len(x)>0 and type(x[0])==type(1)):
                x = [[xx] for xx in x]
            x = list(set(list(itertools.chain.from_iterable([xx for xx in x]))))
            if(len(x)>1):
                try:
                    x.remove(-100)
                    x.remove('O')
                except Exception as e:
                    try:
                        x.remove('O')
                    except:
                        pass
                    pass
            ret_labels.append(x)
        return ret_labels
    def create_multi_class(self, label, sentence):
        # print(label)
        ret_label = [x if type(x)==type(1) else x[0] if(len(x)==1) else re.sub('_[\d]+', '', ('S_' + x[0].replace('B_','').replace('I_', ''))) for x in label]
        ret_label = [re.sub('_[\d]+', '', x) if type(x) == type("xxx") else x for x in ret_label]
        # print('-'*100)
        # print(ret_label)
        # print(sentence)
        # print('_'*100)
        for x in ret_label:
            if(self.labels_frequency.get(x) is None):
                self.labels_frequency[x] = 0
            self.labels_frequency[x] += 1
        return ret_label
    def create_enum_data(self, jsn, global_count_dict, do_upsampling = False):
        turns_once, turn_labels_once, all_triggers = [], [], []
        input_ids, input_labels = [], []
        upsample = False
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            label_index = 0
            turn_dict, turn_same_event_labels = {}, {}
            turns_, turn_labels, turn_triggers = [], [], []
            for event_type in jsn['events'][turns]:#each turn grouped by same event type
                if(event_type=='O' or event_type == "Amend_Action_Data"): 
                    continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                #sort them here to assign in ascending order of indices
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                triggers.sort(key = lambda x: int(x['indices'].split()[0]) if x['indices'].strip()!='' else float('inf'))#sort :D:D
                if(event_type!='O'):
                    # event_type = 'I'
                    # print(">>>>>>", sentence, triggers, len(turn_triggers), turn_triggers)
                    non_zero_counts = sum([1 for tt in triggers if tt['words'].strip()==""])
                    # print(event_type)
                    # if(non_zero_counts>0):
                    #     print('-'*100)
                    #     print(sum([1 for tt in triggers if tt['words'].strip()==""]))
                    #     print(triggers)
                    #     print(sentence)
                    #     print('-'*100)
                    for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                        trigger_vector = ['O'] * len(sentence)
                        turn_triggers.append(trigger['words'])
                        for idx_idx, trigger_index in enumerate(trigger['indices'].split()):
                            # if(global_count_dict.get(event_type) is None):
                            #     global_count_dict[event_type] = 0
                            # if(global_count_dict[event_type]<label_index+1):
                            #     global_count_dict[event_type] = label_index+1
                            try:
                                trigger_index = int(trigger_index)
                                label_ll = ('B' if idx_idx ==0 else 'I') + f"_{event_type}_" + str(label_index)#<<<----- option 2
                                trigger_vector[trigger_index] =  label_ll
                                # if(self.labels_frequency.get(label_ll) is None):
                                #     self.labels_frequency[label_ll] = 0
                                # self.labels_frequency[label_ll] += 1
                            except Exception as e:
                                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", e)
                                pass
                        label_index += 1
                        if(turn_same_event_labels.get(event_type) is None):
                            turn_same_event_labels[event_type] = {'sen': sentence, 'trigger_label':[]}
                        turn_same_event_labels[event_type]['trigger_label'].append(trigger_vector)
                        if(event_type in ["Request_Meeting", "Deliver_Meeting_Data", "Amend_Data", "Request_Data"]):
                            upsample = True
                else: continue
                turn_label = self.merge_labels_at_same_indices(turn_same_event_labels, event_type)
                #print(turn_label)
                input_ids = list(chain(*[jsn['sentences'][i] + ['[SEP]'] for i in range(idx)])) + sentence if idx > 0 else sentence
                input_labels = list(chain(*[[-100]*(len(jsn['sentences'][i])+1) for i in range(idx)])) + turn_label if idx > 0 else turn_label
                assert len(input_ids) == len(input_labels)
                turns_.append(input_ids)
                turn_labels.append({'labels': input_labels, 'event_type': event_type})
            turn_triggers = [x for x in turn_triggers if (x!="" and x!=[""])]
            if(len(turn_triggers)>0):
                all_triggers.append(turn_triggers)
            #print('<><><><><>---->>>', sentence, turn_triggers)
            all_labels = list(zip(*[turn_labels[i]['labels'] for i in range(len(turn_labels))]))
            #print(all_labels)
            new_labels = self.clearTurnLabels(all_labels)
            #print(new_labels)
            #print('*'*100)
            new_labels = self.create_multi_class(new_labels, sentence)
            if(len(turns_)<=0):
                continue
            if(upsample and do_upsampling):
                for _ in range(2):
                    turns_once.append(input_ids)
                    turn_labels_once.append(new_labels)
            turns_once.append(input_ids)
            turn_labels_once.append(new_labels)
        #print(len(all_triggers))
        # print(turns_once, len(turns_once))
        # print(turn_labels_once, len(turn_labels_once))
        # print(all_triggers, len(all_triggers))
        return {'turns':turns_once, 'turn_labels':turn_labels_once, 'turn_triggers': all_triggers}
    def __init__(self, data_addr, tokenizer, max_len = 512, do_upsampling = False, labels_to_ids=None):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = 0
        self.global_count_dict = {}
        self.do_upsampling = do_upsampling
        self.labels_frequency = {}
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            thread_dict = self.create_enum_data(jsn, self.global_count_dict, self.do_upsampling)
            turns, turn_labels = thread_dict['turns'], thread_dict['turn_labels']
            if(len(turns)<=0): continue
            self.samples.extend([{'text': text,'trigger_labels':label, 'trigger_spans': ts,'file_name':os.path.split(file)[-1]} for text, label, ts in zip(turns, turn_labels, thread_dict['turn_triggers'])])
        self.samples = self.samples 
        self.len = len(self.samples)
        # print('Data Loaded. Distribution is as follows: ')
        # print(self.global_count_dict) 
        if(labels_to_ids is not None):
            self.labels_to_ids = labels_to_ids
        else:
            self.labels_to_ids = {'O':0}
            for label in self.labels_frequency:
                label_counts = self.labels_frequency[label]
                #for i in range(label_counts):
                if(label==-100 or self.labels_to_ids.get(label) is not None): continue
                self.labels_to_ids[label] = len(self.labels_to_ids)
        self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        # print(">>>>>>>", self.labels_frequency, "<<<<<<")
    def get_labels_to_ids(self):
        return self.labels_to_ids, self.ids_to_labels
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
    def get_loss_mask(self, sep_indices, current_label, pad_starts_from):
        loss_mask = [1]*len(current_label)
        loss_mask[0] = 0#for [CLS]
        if(len(sep_indices)>0):
            last_sep_index = sep_indices[-1]
            while(last_sep_index>=0):
                loss_mask[last_sep_index] = 0
                last_sep_index -= 1
        pad_starts_from-=1#including [SEP]
        while(pad_starts_from<self.max_len):
            loss_mask[pad_starts_from] = 0
            pad_starts_from += 1
        return np.array(loss_mask)
    def __getitem__(self, index):
        sample = self.samples[index]
        complete_email, complete_trigger_labels, file_name, trigger_spans = sample['text'], sample['trigger_labels'], sample['file_name'], sample['trigger_spans']#, sample['event_type']
        complete_email = [word if (word!=[''] and word!='')  else '*' for word in complete_email]
        # print([(w, l) for w, l in zip(complete_email, complete_trigger_labels)])
        # print('-'*100)
        #complete_trigger_labels = [[x] if type(x)==int else x for x in complete_trigger_labels]
        #complete_trigger_labels = self.order_labels(complete_trigger_labels)
        # print(trigger_spans)
        # print(complete_trigger_labels)
        # print(complete_email)
        # print('-'*100)
        encoding = self.tokenizer(complete_email, is_split_into_words = True, return_offsets_mapping = True)
        encoded_labels = list([-100]*len(encoding["offset_mapping"]))
        labels_ = []
        for lbl in complete_trigger_labels:
            #word_labels = []
            #for l in lbl:
            #        enc_lbl = self.labels_to_ids[l] if self.labels_to_ids.get(l) is not None else -100
            #        word_labels.append(enc_lbl)
            labels_.append(self.labels_to_ids.get(lbl))
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if(mapping[0]!=0 and mapping[1]!=0):
                last_label = self.labels_to_ids[self.ids_to_labels[labels_[i-1]]] if self.ids_to_labels.get(labels_[i-1]) is not None else -100 #[self.labels_to_ids[self.ids_to_labels[x]]  if self.ids_to_labels.get(x) is not None else -100 for x in labels_[i-1]]
                encoded_labels[idx] = last_label #assign I- of the "last_event" labels to sub-words 
            elif(mapping[0]==0 and mapping[1]!=0):
                encoded_labels[idx]=labels_[i]
                i+=1
        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + encoding['input_ids'][-511:] if len(encoding['input_ids'])>512 else encoding['input_ids']
        labels = [-100] + encoded_labels[-511:] if len(encoded_labels)>512 else encoded_labels#cls has -100 so -511
        attention_mask = encoding['attention_mask'][-512:]
        attention_mask += [0]*(self.max_len - len(attention_mask))
        assert len(input_ids) == len(labels)
        input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(self.max_len - len(input_ids))
        pad_starts_from = len(labels)
        labels += [-100]*(self.max_len - len(labels))#fail safe for padding?
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'], sep_indices = self.nullifyHistory(input_ids, labels)
        loss_mask = self.get_loss_mask(sep_indices, item['labels'], pad_starts_from)
        for idx, input_id in enumerate(input_ids):
            if(input_id in [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)]):
                item['labels'][idx] = -100#torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)        
        item['labels'] = torch.tensor(item['labels'], dtype=torch.long)
        item['input_ids'] = torch.tensor(input_ids)
        item['attention_mask'] = torch.tensor(attention_mask)
        item['token_type_ids'] = [0]*self.max_len
        item['loss_mask'] = torch.tensor(loss_mask)
        item["offset_mapping"] = torch.tensor( item["offset_mapping"].tolist()[-512:] + [(-1, -1)]*(self.max_len - len(item["offset_mapping"])))
        #print(self.tokenizer.decode(self.tokenizer('[SEP]'.join(trigger_spans))['input_ids']))
        item['trigger_span'] = torch.tensor(self.tokenizer('[SEP]'.join(trigger_spans), pad_to_max_length=True, max_length=512)['input_ids'])
        #x = torch.tensor([101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # print("Complete email: ", complete_email, '\nTS: ', trigger_spans)
        # print("TS: ", item['trigger_span'])
        # if(sum(item['trigger_span']) == 101+102):
        #     tokenized_text = self.tokenizer.convert_ids_to_tokens(item['input_ids'][item['loss_mask']==1])
        #     print("check this: ", trigger_spans)
        #     text_label_pair = [(tt, self.ids_to_labels[lbl] if lbl!=-100 else -100) for tt, lbl in zip(tokenized_text, item['labels'][item['loss_mask']==1].tolist())]
        #     print("TL: ", text_label_pair)
            #print("labels: ", item['labels'].tolist())
        #print(item["offset_mapping"].shape)
        del item["token_type_ids"]
        #del item['offset_mapping']
        return item
    def __len__(self):
        return self.len



class seq_tagger_trigger_multi_class_single_turn(Dataset):
    def merge_labels_at_same_indices(self, turn_same_event_labels, event_name):
        tl = turn_same_event_labels[event_name]['trigger_label']
        if len(tl) > 1:
            multi_label = list(zip(*tl))
            for idx, label in enumerate(multi_label):
                uniques = set(label)
                if(len(uniques)>1):uniques.discard('O')
                multi_label[idx] = list(uniques)
        else:
            multi_label = [[x] for x in tl[0]]
        return multi_label
    def extract_event_type(self, label):
        return label.replace("B-", '').replace("I-", '').strip()
    def clearTurnLabels(self, turn_labels):
        ret_labels = []
        for x in turn_labels:#a multu-label tuple
            if(len(x)>0 and type(x[0])==type(1)):
                x = [[xx] for xx in x]
            x = list(set(list(itertools.chain.from_iterable([xx for xx in x]))))
            if(len(x)>1):
                try:
                    x.remove(-100)
                    x.remove('O')
                except Exception as e:
                    try:
                        x.remove('O')
                    except:
                        pass
                    pass
            ret_labels.append(x)
        return ret_labels
    def create_multi_class(self, label, sentence):
        # print(label)
        ret_label = [x if type(x)==type(1) else x[0] if(len(x)==1) else re.sub('_[\d]+', '', ('S_' + x[0].replace('B_','').replace('I_', ''))) for x in label]
        ret_label = [re.sub('_[\d]+', '', x) if type(x) == type("xxx") else x for x in ret_label]
        # print('-'*100)
        # print(ret_label)
        # print(sentence)
        # print('_'*100)
        for x in ret_label:
            if(self.labels_frequency.get(x) is None):
                self.labels_frequency[x] = 0
            self.labels_frequency[x] += 1
        return ret_label
    def create_enum_data(self, jsn, global_count_dict, do_upsampling = False):
        turns_once, turn_labels_once, all_triggers = [], [], []
        input_ids, input_labels = [], []
        upsample = False
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            label_index = 0
            turn_dict, turn_same_event_labels = {}, {}
            turns_, turn_labels, turn_triggers = [], [], []
            for event_type in jsn['events'][turns]:#each turn grouped by same event type
                if(event_type=='O' or event_type == "Amend_Action_Data"): 
                    continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                #sort them here to assign in ascending order of indices
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                self.all_triggers.extend(triggers)
                triggers.sort(key = lambda x: int(x['indices'].split()[0]) if x['indices'].strip()!='' else float('inf'))#sort :D:D
                if(event_type!='O'):
                    # event_type = 'I'
                    # print(">>>>>>", sentence, triggers, len(turn_triggers), turn_triggers)
                    non_zero_counts = sum([1 for tt in triggers if tt['words'].strip()==""])
                    # print(event_type)
                    # if(non_zero_counts>0):
                    #     print('-'*100)
                    #     print(sum([1 for tt in triggers if tt['words'].strip()==""]))
                    #     print(triggers)
                    #     print(sentence)
                    #     print('-'*100)
                    for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                        trigger_vector = ['O'] * len(sentence)
                        turn_triggers.append(trigger['words'])
                        for idx_idx, trigger_index in enumerate(trigger['indices'].split()):
                            # if(global_count_dict.get(event_type) is None):
                            #     global_count_dict[event_type] = 0
                            # if(global_count_dict[event_type]<label_index+1):
                            #     global_count_dict[event_type] = label_index+1
                            try:
                                trigger_index = int(trigger_index)
                                label_ll = ('B' if idx_idx ==0 else 'I') + f"_{event_type}_" + str(label_index)#<<<----- option 2
                                trigger_vector[trigger_index] =  label_ll
                                # if(self.labels_frequency.get(label_ll) is None):
                                #     self.labels_frequency[label_ll] = 0
                                # self.labels_frequency[label_ll] += 1
                            except Exception as e:
                                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", e)
                                pass
                        label_index += 1
                        if(turn_same_event_labels.get(event_type) is None):
                            turn_same_event_labels[event_type] = {'sen': sentence, 'trigger_label':[]}
                        turn_same_event_labels[event_type]['trigger_label'].append(trigger_vector)
                        if(event_type in ["Request_Meeting", "Deliver_Meeting_Data", "Amend_Data", "Request_Data"]):
                            upsample = True
                else: continue
                turn_label = self.merge_labels_at_same_indices(turn_same_event_labels, event_type)
                #print(turn_label)
                input_ids =  sentence
                input_labels = turn_label
                assert len(input_ids) == len(input_labels)
                turns_.append(input_ids)
                turn_labels.append({'labels': input_labels, 'event_type': event_type})
            turn_triggers = [x for x in turn_triggers if (x!="" and x!=[""])]
            if(len(turn_triggers)>0):
                all_triggers.append(turn_triggers)
            #print('<><><><><>---->>>', sentence, turn_triggers)
            all_labels = list(zip(*[turn_labels[i]['labels'] for i in range(len(turn_labels))]))
            #print(all_labels)
            new_labels = self.clearTurnLabels(all_labels)
            #print(new_labels)
            #print('*'*100)
            new_labels = self.create_multi_class(new_labels, sentence)
            if(len(turns_)<=0):
                continue
            if(upsample and do_upsampling):
                for _ in range(2):
                    turns_once.append(input_ids)
                    turn_labels_once.append(new_labels)
            turns_once.append(input_ids)
            turn_labels_once.append(new_labels)
        #print(len(all_triggers))
        # print(turns_once, len(turns_once))
        # print(turn_labels_once, len(turn_labels_once))
        # print(all_triggers, len(all_triggers))
        return {'turns':turns_once, 'turn_labels':turn_labels_once, 'turn_triggers': all_triggers}
    def __init__(self, data_addr, tokenizer, max_len = 512, do_upsampling = False, labels_to_ids=None):
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = 0
        self.global_count_dict = {}
        self.do_upsampling = do_upsampling
        self.labels_frequency = {}
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            thread_dict = self.create_enum_data(jsn, self.global_count_dict, self.do_upsampling)
            turns, turn_labels = thread_dict['turns'], thread_dict['turn_labels']
            if(len(turns)<=0): continue
            self.samples.extend([{'text': text,'trigger_labels':label, 'trigger_spans': ts,'file_name':os.path.split(file)[-1]} for text, label, ts in zip(turns, turn_labels, thread_dict['turn_triggers'])])
        self.samples = self.samples 
        self.len = len(self.samples)
        self.all_triggers = []#store all the triggers for debuging purpose
        # print('Data Loaded. Distribution is as follows: ')
        # print(self.global_count_dict) 
        if(labels_to_ids is not None):
            self.labels_to_ids = labels_to_ids
        else:
            self.labels_to_ids = {'O':0}
            for label in self.labels_frequency:
                label_counts = self.labels_frequency[label]
                #for i in range(label_counts):
                if(label==-100 or self.labels_to_ids.get(label) is not None): continue
                self.labels_to_ids[label] = len(self.labels_to_ids)
        self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        # print(">>>>>>>", self.labels_frequency, "<<<<<<")
    def print_all_triggers(self):
        print(self.all_triggers)
    def get_labels_to_ids(self):
        return self.labels_to_ids, self.ids_to_labels
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
    def get_loss_mask(self, sep_indices, current_label, pad_starts_from):
        loss_mask = [1]*len(current_label)
        loss_mask[0] = 0#for [CLS]
        if(len(sep_indices)>0):
            last_sep_index = sep_indices[-1]
            while(last_sep_index>=0):
                loss_mask[last_sep_index] = 0
                last_sep_index -= 1
        pad_starts_from-=1#including [SEP]
        while(pad_starts_from<self.max_len):
            loss_mask[pad_starts_from] = 0
            pad_starts_from += 1
        return np.array(loss_mask)
    def __getitem__(self, index):
        sample = self.samples[index]
        complete_email, complete_trigger_labels, file_name, trigger_spans = sample['text'], sample['trigger_labels'], sample['file_name'], sample['trigger_spans']#, sample['event_type']
        complete_email = [word if (word!=[''] and word!='')  else '*' for word in complete_email]
        # print([(w, l) for w, l in zip(complete_email, complete_trigger_labels)])
        # print('-'*100)
        #complete_trigger_labels = [[x] if type(x)==int else x for x in complete_trigger_labels]
        #complete_trigger_labels = self.order_labels(complete_trigger_labels)
        # print(trigger_spans)
        # print(complete_trigger_labels)
        # print(complete_email)
        # print('-'*100)
        encoding = self.tokenizer(complete_email, is_split_into_words = True, return_offsets_mapping = True)
        encoded_labels = list([-100]*len(encoding["offset_mapping"]))
        labels_ = []
        for lbl in complete_trigger_labels:
            #word_labels = []
            #for l in lbl:
            #        enc_lbl = self.labels_to_ids[l] if self.labels_to_ids.get(l) is not None else -100
            #        word_labels.append(enc_lbl)
            labels_.append(self.labels_to_ids.get(lbl))
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if(mapping[0]!=0 and mapping[1]!=0):
                last_label = self.labels_to_ids[self.ids_to_labels[labels_[i-1]]] if self.ids_to_labels.get(labels_[i-1]) is not None else -100 #[self.labels_to_ids[self.ids_to_labels[x]]  if self.ids_to_labels.get(x) is not None else -100 for x in labels_[i-1]]
                encoded_labels[idx] = last_label #assign I- of the "last_event" labels to sub-words 
            elif(mapping[0]==0 and mapping[1]!=0):
                encoded_labels[idx]=labels_[i]
                i+=1
        input_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + encoding['input_ids'][-511:] if len(encoding['input_ids'])>512 else encoding['input_ids']
        labels = [-100] + encoded_labels[-511:] if len(encoded_labels)>512 else encoded_labels#cls has -100 so -511
        attention_mask = encoding['attention_mask'][-512:]
        attention_mask += [0]*(self.max_len - len(attention_mask))
        assert len(input_ids) == len(labels)
        input_ids += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(self.max_len - len(input_ids))
        pad_starts_from = len(labels)
        labels += [-100]*(self.max_len - len(labels))#fail safe for padding?
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'], sep_indices = self.nullifyHistory(input_ids, labels)
        loss_mask = self.get_loss_mask(sep_indices, item['labels'], pad_starts_from)
        for idx, input_id in enumerate(input_ids):
            if(input_id in [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token), self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)]):
                item['labels'][idx] = -100#torch.tensor([-100]*len(self.labels_to_ids), dtype=torch.float)        
        item['labels'] = torch.tensor(item['labels'], dtype=torch.long)
        item['input_ids'] = torch.tensor(input_ids)
        item['attention_mask'] = torch.tensor(attention_mask)
        item['token_type_ids'] = [0]*self.max_len
        item['loss_mask'] = torch.tensor(loss_mask)
        item["offset_mapping"] = torch.tensor( item["offset_mapping"].tolist()[-512:] + [(-1, -1)]*(self.max_len - len(item["offset_mapping"])))
        #print(self.tokenizer.decode(self.tokenizer('[SEP]'.join(trigger_spans))['input_ids']))
        item['trigger_span'] = torch.tensor(self.tokenizer('[SEP]'.join(trigger_spans), pad_to_max_length=True, max_length=512)['input_ids'])
        #x = torch.tensor([101, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # print("Complete email: ", complete_email, '\nTS: ', trigger_spans)
        # print("TS: ", item['trigger_span'])
        # if(sum(item['trigger_span']) == 101+102):
        #     tokenized_text = self.tokenizer.convert_ids_to_tokens(item['input_ids'][item['loss_mask']==1])
        #     print("check this: ", trigger_spans)
        #     text_label_pair = [(tt, self.ids_to_labels[lbl] if lbl!=-100 else -100) for tt, lbl in zip(tokenized_text, item['labels'][item['loss_mask']==1].tolist())]
        #     print("TL: ", text_label_pair)
            #print("labels: ", item['labels'].tolist())
        #print(item["offset_mapping"].shape)
        del item["token_type_ids"]
        #del item['offset_mapping']
        return item
    def __len__(self):
        return self.len