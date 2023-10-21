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
from genTemplates import gen_template, template_function_call
class gen_with_templates_dataset(Dataset):
    def create_enum_data(self, jsn, global_count_dict, include_history = True, file_name = None):
        turns_once, turn_labels_once, all_triggers, prompt_once = [], [], [], []
        input_ids, input_labels, filled_template = [], [], ''
        for idx, (turns, sentence) in enumerate(zip(jsn['events'], jsn['sentences'])):
            turn_dict, turn_same_event_templates = {}, {}
            turns_, turn_labels, turn_triggers, prompt_ = [], [], [], []
            for event_type in jsn['events'][turns]:#each turn is grouped by same event type
                #print(f'>{event_type}<')
                if(event_type=='O' or event_type == "Amend_Action_Data" or event_type.strip() == ""): 
                    #in the current set of experiments, we ignore AAD due to their low frequency, and "" or O is ignored because it is not an event
                    continue
                labels, triggers, metaSRs = jsn['events'][turns][event_type]['labels'], jsn['events'][turns][event_type]['triggers'], jsn['events'][turns][event_type]['extras']
                #sort them here to assign in ascending order of indices
                triggers = [ast.literal_eval(x) if x.strip()!='' else {'words':"", 'indices': ''} for x in triggers]
                original_triggers = copy.deepcopy(triggers)
                triggers.sort(key = lambda x: int(x['indices'].split()[0]) if x['indices'].strip()!='' else float('inf'))#sort
                new_labels = [None]*(len(triggers))
                for x_idx, (trigger, label) in enumerate(zip(original_triggers, labels)):
                    for sorted_idx, sorted_trigger in enumerate(triggers):
                        if(sorted_trigger==trigger):
                            new_labels[sorted_idx] = label
                for x in new_labels:
                    assert x is not None
                labels = new_labels
                if(event_type!='O'):
                    if(self.labels_frequency.get(event_type) is None):
                        self.labels_frequency[event_type] = 0
                    self.labels_frequency[event_type] += 1
                    non_zero_counts = sum([1 for tt in triggers if tt['words'].strip()==""])
                    for repeat_index, (label, trigger, metaSR) in enumerate(zip(labels, triggers, metaSRs)):#data about events of same type
                        filled_template = gen_template(trigger, label, sentence, event_type, metaSR)
                        trigger_vector = ['O'] * len(sentence)
                        turn_triggers.append(trigger['words'])
                        if(turn_same_event_templates.get(event_type) is None):
                            turn_same_event_templates[event_type] = {'templates':[]}
                        turn_same_event_templates[event_type]['templates'].append(filled_template)
                else: continue
                #input_ids = list(chain(*[jsn['sentences'][i] + [f'{self.tokenizer.eos_token}'] for i in range(idx)])) + sentence if idx > 0 else sentence
                #input_ids = list(chain(*[jsn['sentences'][i] + ['[CONTEXT]'] for i in range(idx)])) + [f'{self.tokenizer.eos_token}'] + sentence if idx > 0 else sentence
                #print(sentence, file_name)
                #print(len(jsn['sentences']), idx)
                ###to use current email only use the sentence only###
                if(include_history):
                    input_ids = list(chain(*[jsn['sentences'][i] + ['[CONTEXT]'] for i in range(idx)])) + sentence if idx > 0 else sentence
                else:
                    input_ids = sentence
                prompt_ids = ["\nContext:\n"] + list(chain(*[jsn['sentences'][i] + ['\n'] for i in range(idx)])) + ["Current Email:\n"] + sentence if idx > 0 else ["\nContext:\nCurrent Email:\n"] + sentence
            if(event_type.strip() == "O" or event_type.strip() == ""):
                continue#to handle the last turns which are OO
            turns_.append(input_ids)
            prompt_.append(prompt_ids)
            turns_once.extend(turns_)#okay
            prompt_once.extend(prompt_)
            turn_labels_once.append(turn_same_event_templates)
        return {'turns':turns_once, 'turn_labels':turn_labels_once, 'prompts':prompt_once}
    def __init__(self, data_addr, tokenizer, max_len = 512, include_history = True, labels_to_ids=None):
        self.total_samples = 0
        self.samples = []
        self.data_addr = data_addr
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = 0
        self.global_count_dict = {}
        self.include_history = include_history
        self.labels_frequency = {'Request_Meeting': 0, 'Request_Data': 0, 'Request_Action': 0, 'Request_Action_Data': 0, 'Request_Meeting_Data': 0, 'Deliver_Data': 0, 'Deliver_Action_Data': 0, 'Deliver_Meeting_Data': 0, 'Amend_Data': 0, 'Amend_Meeting_Data': 0}
        self.stats = {'total_max_length':0, 'total_avg_length':0, 'num_greater_than_max_length':0, 'after_concat_max':0, 'after_concat_avg':0, 'after_concat_max_length':0, 'label_max_length':0, 'label_avg_length':0, 'label_greater_than_max_len':0, 'largest_input':"", "largest_label":""}
        files = glob(os.path.join(data_addr, "*.json"))
        for file in files:
            f = open(file, 'r')
            jsn = json.load(f)
            thread_dict = self.create_enum_data(jsn, self.global_count_dict, self.include_history, file_name = file)
            turns, turn_labels, prompts = thread_dict['turns'], thread_dict['turn_labels'], thread_dict["prompts"]
            if(len(turns)<=0): 
                continue
            self.samples.extend([{'text': text,'trigger_labels':label, 'prompt': prompt ,'file_name':os.path.split(file)[-1]} for text, label, prompt in zip(turns, turn_labels, prompts)])
        self.len = len(self.samples)
        if(labels_to_ids is not None):
            self.labels_to_ids = labels_to_ids
        else:
            self.labels_to_ids = {'O':0}
            for label in self.labels_frequency:
                label_counts = self.labels_frequency[label]
                if(label==-100 or self.labels_to_ids.get(label) is not None): continue
                self.labels_to_ids[label] = len(self.labels_to_ids)
        self.ids_to_labels = dict((v,k) for k, v in self.labels_to_ids.items())
        self.prompt_data = []
    def dump_prompt_data(self, type_):
        with open(f'./{type_}_prompt_data.json', 'w') as f:
            json.dump(self.prompt_data, f, indent = 4)
    def get_labels_to_ids(self):
        return self.labels_to_ids, self.ids_to_labels
    def get_all_templates(self, current_template, complete_emails, include_blanks = True, file_name=None):
        #get the most frequest meta semantic roles
        max_requested_attribs = {'Amend_Data': "Update", "Deliver_Action_Data": "Positive", "Deliver_Data": "Positive", "Deliver_Meeting_Data": "Positive", "Request_Action_Data": "Action Description", "Request_Data": "Data Value", "Request_Meeting_Data": "Meeting Date"}
        events_covered = list(set(current_template.keys()))
        template, label_template, template_for_input = "", "", ""
        for event in list(self.labels_frequency.keys()):#ensure same order always
            del_req_att = max_requested_attribs[event] if max_requested_attribs.get(event) is not None else ''
            template_for_input += self.tokenizer.begin_of_template + " " + re.sub('[ \t]+',' ',template_function_call[event.replace('_', " ")]({}, del_req_att, 'trigger').get_filled_template()) + " " + self.tokenizer.end_of_template + " " + self.tokenizer.eos_token + " "
            if(event in events_covered):
                new_templates = ' '.join([self.tokenizer.begin_of_template + " " + x + " " + self.tokenizer.end_of_template + " " + self.tokenizer.eos_token for x in (current_template[event]['templates'])])#join doesn't covers the last one
                label_template += " " + new_templates
            else:
                if(include_blanks or current_template=={}):
                    #request a blank one
                    del_req_att = max_requested_attribs[event] if max_requested_attribs.get(event) is not None else ''
                    new_templates = self.tokenizer.begin_of_template + " " + re.sub('[ \t]+',' ',template_function_call[event.replace('_', " ")]({}, del_req_att, 'trigger').get_filled_template()) + " " + self.tokenizer.end_of_template + " " + self.tokenizer.eos_token
                else:
                    new_templates = ''
            template += " " + new_templates 
        template, label_template, template_for_input = re.sub('[ \t]+', " ", template), re.sub('[ \t]+', " ", label_template), re.sub('[ \t]+', " ", template_for_input)
        template, label_template, template_for_input = template.strip(), label_template.strip(), template_for_input.strip()
        #for debugging
        if(label_template==""):
            print('complete_emails: ', complete_emails)
            print('File name: ', file_name)
            print('current_template: ', current_template)
            print('*'*100)
        template, label_template, template_for_input = (template[:-4] if(template[-4:].strip()==self.tokenizer.eos_token) else template).strip(), (label_template[:-4] if(label_template[-4:].strip()!="" and label_template[-4:].strip()==self.tokenizer.eos_token) else label_template).strip(), (template_for_input[:-4] if(template_for_input[-4:].strip()==self.tokenizer.eos_token) else template_for_input).strip()
        return template, label_template, template_for_input
    def __getitem__(self, index):
        sample = self.samples[index]
        complete_email, complete_trigger_labels, file_name, prompt = sample['text'], sample['trigger_labels'], sample['file_name'], sample['prompt']
        
        complete_email = [word if (word!=[''] and word!='')  else '*' for word in complete_email]
        prompt = [word if (word!=[''] and word!='')  else '*' for word in prompt]
        template, label_template, template_for_input = self.get_all_templates(complete_trigger_labels, complete_email, file_name = file_name)
        template = template[:-4] if template[-4:] == f'{self.tokenizer.eos_token}' else template
        
        prompt_data = ' '.join(prompt).strip() #+ '\n' + template_for_input.replace(self.tokenizer.eos_token, "\n").replace(self.tokenizer.begin_of_template, "").replace(self.tokenizer.end_of_template, "")
        prompt_data += "\nThe outputs are:\n" + label_template.replace(self.tokenizer.eos_token, "\n").replace(self.tokenizer.begin_of_template, "").replace(self.tokenizer.end_of_template, "")
        prompt_data = re.sub('[ \t]', ' ', prompt_data).strip()
        
        self.prompt_data.append({"prompt": prompt_data, "email_body":' '.join(complete_email)})
        
        complete_email = ' '.join(complete_email)  + self.tokenizer.eos_token  + self.tokenizer.eos_token + template_for_input#+ template#.split()
        encoding = self.tokenizer(complete_email, pad_to_max_length=True, max_length = self.max_len, truncation=True)
        self.total_samples += 1
        ###LABELS
        label_template = label_template[:-4] if (label_template.strip()!="" and label_template[-4:] == f'{self.tokenizer.eos_token}') else label_template
        label = self.tokenizer(label_template, pad_to_max_length=True, max_length = self.max_len, truncation=True)
        
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label['input_ids'], dtype=torch.long)
        item['input_ids'] = torch.tensor(encoding['input_ids'])
        item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.bool)
        item['decoder_attention_mask'] = torch.tensor(label['attention_mask'], dtype=torch.bool)
        item['file_name'] = torch.tensor(self.tokenizer(file_name, pad_to_max_length=True, max_length = self.max_len, truncation=True)['input_ids'])
        return item
    def __len__(self):
        return self.len

def my_collate(batch):
    input_token_ids = torch.stack([torch.LongTensor(ex['input_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['attention_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['labels']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['decoder_attention_mask']) for ex in batch])

    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'file_names': torch.stack([torch.LongTensor(ex['file_name']) for ex in batch])
        }

if __name__ == '__main__':
    import re
    from gen_data_loader import gen_with_templates_dataset, my_collate
    from transformers import BartTokenizer, BartConfig
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', truncation_side = "right", add_prefix_space=True)
    tokenizer.add_tokens(['[CONTEXT]', '[EOT]', '[BOT]'])
    tokenizer.end_of_template = '[EOT]'
    tokenizer.begin_of_template = '[BOT]'
    x = gen_with_templates_dataset('./data/train', tokenizer, 1024)
    for d in x:
        pass
    x.dump_prompt_data('train')
    x = gen_with_templates_dataset('./data/dev', tokenizer, 1024)
    for d in x:
        pass
    x.dump_prompt_data('dev')
    x = gen_with_templates_dataset('./data/test', tokenizer, 1024)
    for d in x:
        pass
    x.dump_prompt_data('test')
