import os 
import argparse 
import torch 
import logging 
import json 
from difflib import SequenceMatcher
from torch.nn import functional as F 
import pytorch_lightning as pl 
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from genTemplates import *
from network import BartGen
from collections import defaultdict 
import re, torch
import evaluate
logger = logging.getLogger(__name__)
from tqdm import tqdm
all_templates = "[EOT] [BOT] Event Request Meeting is triggered by | trigger | where , | Meeting Name | is requested among | Meeting Members | at | Meeting Time | on | Meeting Date | at | Meeting Location | to discuss | Meeting Agenda | Event Request Data is triggered by | trigger | where , | Context: Data idString | of | Context: Data Type | by | Context: Data Owner | is requested from | Context: Request members | to be delivered at | Context: Request Time | on | Context: Request Date | Event Request Action is triggered by | trigger | where , | Action Description | is requested from | Action Members | at | Action Time | on | Action Date | Event Request Action Data is triggered by | trigger | where , Action Description is requested for | Context: Action Description | by | Context: Action Members | at | Context: Action Time | from | Context: Request Members | Event Request Meeting Data is triggered by | trigger | where , Date is requested for | Context: Meeting Name | among | Context: Meeting Members | at | Context: Meeting Time | at | Context: Meeting Location | to discuss | Context: Meeting Agenda | from | Context: Request Members | Event Deliver Data is triggered by | trigger | where , | Data idString |, | Data Value | of | Data Type | is or will be delivered to | Deliver Members | at | Deliver Time | on | Deliver Date | Event Deliver Action Data is triggered by | trigger | where , | Action Description | is or will be performed by | Action Members | at | Action Time | on | Action Date | Event Deliver Meeting Data is triggered by | trigger | where , | Meeting Name | is or will be attended by | Meeting Members | at | Meeting Time | on | Meeting Date | at | Meeting Location | to discuss | Meeting Agenda | Event Amend Data is triggered by | trigger | where , For | Context: Data idString |, | Context: Data Value | is or requested to be updated to | Revision: Data Value | from | Context: Amend Members | at | Context: Amend Time | on | Context: Amend Date | Event Amend Meeting Data is triggered by | trigger | where , For | Context: Meeting Name | among | Context: Meeting Members | at | Context: Meeting Time | on | Context: Meeting Date | at | Context: Meeting Location | to discuss | Context: Meeting Agenda |, date is or requested to be updated to | Revision: Meeting Date | from | Context: Amend Members |"

MAX_LENGTH=512

class GenIEModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams = args 
        self.config=BartConfig.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', truncation_side = "right", add_prefix_space=True)
        self.tokenizer.add_tokens(['[CONTEXT]', '[EOT]', '[BOT]'])
        self.tokenizer.end_of_template = '[EOT]'
        self.tokenizer.begin_of_template = '[BOT]'     
        self.best_metric = -1
        self.best_json_file = None
        if self.hparams.model=='gen':
            self.model = BartGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings() 
        elif self.hparams.model == 'constrained-gen':
            self.model = BartConstrainedGen(self.config, self.tokenizer)
            self.model.resize_token_embeddings() 
        else:
            raise NotImplementedError
        self.eval_dict = []
        self.step_count = 0
    def forward(self, inputs):    
        return self.model(**inputs)
    def training_step(self, batch, batch_idx):
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],   
                    "task": 0 
                }
        outputs = self.model(**inputs)
        loss = outputs[0]
        loss = torch.mean(loss)
        log = {
            'train/loss': loss, 
        } 
        return {
            'loss': loss, 
            'log': log 
        }
    def extract_args(self, template, delimiter = "|"):
        #template = re.sub('[ \t]+', ' ', template.replace(',', ' ,').replace('?', ' ?').strip())
        #chuityapa
        #print('template>>>>>', template)
        template = template.split()
        extracted_args = []
        idx = 0
        while(idx<len(template)):
            #print(template[idx], template[idx]==delimiter)
            arg = []
            if(template[idx]==delimiter):
                idx += 1
                while(idx<len(template) and template[idx]!=delimiter):
                    #print(template[idx], template[idx]!=delimiter)
                    arg.append(template[idx])
                    idx += 1
            #print(arg)
            if(len(arg)>0):
                extracted_args.append(re.sub('[ \t]+', ' ', ' '.join(arg).replace('|', "")))
            idx+=1
        return extracted_args
    def extract_event_trigger_only(self, template):
        ret_outputs = []
        import re
        #print('>>>>>>>>>>>>>>>>>', generated_templates)
        all_templates = generated_templates.split(self.tokenizer.end_of_template)
        for template in all_templates:
            if(template.strip()==""):
                continue
            template = template.replace(self.tokenizer.begin_of_template, '').strip()
            event_template = r'Event (.+?) is triggered by \| (.+?) \| where, '
            event_trigger = re.match(event_template, template)
            try:
                event, trigger = event_trigger.groups()
            except Exception as e:
                event, trigger = "", ""
                #print('Exception at 97: ', e)
            ret_outputs.append({'trigger': trigger, "event_type": event})
        return ret_outputs
    def validation_step(self,batch, batch_idx):
        # print('*'*100)
        # print('Len: ', len(self.eval_dict))
        # print('*'*100)
        # print(batch)
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],  
                    "task" :0,   
                }
        file_names = batch["file_names"]
        outputs = self.model(**inputs)
        #test = self.model.generate(inputs["input_ids"])
        logits = outputs[1]
        #print(logits)
        probs = F.softmax(logits, dim=-1)
        #print(probs, probs.shape)
        argmax_op = torch.argmax(probs, dim=-1)


        # gold_templates = self.tokenizer.batch_decode(inputs['decoder_input_ids'], skip_special_tokens=True)
        # pred_templates = self.tokenizer.batch_decode(argmax_op, skip_special_tokens=True)
        # input_emails = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)

        # file_names = self.tokenizer.batch_decode(file_names, skip_special_tokens=True)
        # gt = self.tokenizer.convert_ids_to_tokens(inputs['decoder_input_ids'][0], skip_special_tokens=True)
        # print(gt)
        # print(gold_templates)
        # xxx
        #assert len(input_email) == len(file_names) == len(gold_templates) == len(pred_templates)
        #assert len(gold_templates) == len(pred_templates)
        #assert False
        gts, pts, ips, fns, full_inputs, full_labels, full_predicted_labels = [], [], [], [], [], [], []
        for _, (gt, pt, email, fn) in tqdm(enumerate(zip(inputs['decoder_input_ids'], argmax_op, inputs['input_ids'], file_names)), total = len(file_names), desc = "Constrained Decoding"):
            gold_template = self.tokenizer.decode(gt, skip_special_tokens = True)
            pred_template = self.tokenizer.decode(pt, skip_special_tokens = True)#argmax_op <- probs <- logits <- outputs <- inputs
            #input_ids -> email

            # print("Pred_templates:", pred_template)
            # print("Email: ", self.tokenizer.decode(email.tolist()))
            # email = 
            # print("PT:", pt)
            gen_ids = self.model.generate(email.unsqueeze(0), force_words_ids=[email.tolist() + self.tokenizer(all_templates)['input_ids']], max_length = 1000)
            # print(gen_ids, gen_ids[0], gen_ids.tolist())
            constrained_gen = self.tokenizer.decode(gen_ids.tolist()[0], skip_special_tokens = True)
            # print("Generate: ", constrained_gen)
            # xxx
            if(gold_template.strip()==""):
                continue
            ip = self.tokenizer.decode(email, skip_special_tokens = True)
            gts.append(gold_template)
            pts.append(constrained_gen)
            # pts.append(pred_template)
            ips.append(ip)
            fns.append(self.tokenizer.decode(fn, skip_special_tokens = True))
            full_inputs.append(self.tokenizer.decode(email))
            full_labels.append(gt.tolist())
            full_predicted_labels.append(pt.tolist())
        # gold_templates = [x for x in gold_templates if x.strip()!=""]
        # pred_templates = [x for x in pred_templates if x.strip()!=""]
        gold_templates, pred_templates, input_email, file_names = gts, pts, ips, fns
        # print('*'*100)
        # print('Input and Output Preview: ')
        # print('Input: ', input_email[0])
        # print('Label: ', gold_templates[0])
        # print('Prediction: ', pred_templates[0])
        # print('File name: ', file_names[0])
        # print('*'*100)
        assert len(input_email) == len(gold_templates) == len(pred_templates) == len(file_names)
        for gt, pt, email, file_name, full_input, full_label, full_pred_label in zip(gold_templates, pred_templates, input_email, file_names, full_inputs, full_labels, full_predicted_labels):
            full_label = self.tokenizer.decode(full_label)
            full_pred_label = self.tokenizer.decode(full_pred_label)
            #print('before', pt)
            pt = re.sub("[ \t]+", " ", pt.replace(",", " ,").replace(".", " .").replace("?", " ?").replace("'", " '"))
            #print('after', pt)
            gt = re.sub("[ \t]+", " ", gt.replace(",", " ,").replace(".", " .").replace("?", " ?").replace("'", " '"))
            #gt_tok = self.tokenizer.batch_decode(gt, skip_special_tokens=True)
            print(gt)
            ops = self.extract_args_from_template(gt)
            print('------------------')
            print(ops)
            assert len(ops)>0
            ########################
            #print('pt>>>', pt, '<<<pt')
            #print(self.extract_args_from_template(pt))
            #print('gt>>>', gt, '<<<gt')
            ########################
            try:
                pops = self.extract_args_from_template(pt)
                assert len(pops)>0
            except Exception as e:
                #print('Exception at 142: ', e)
                pass
                #pops = [{'event_type': "", "event_trigger":""}]
            event_dict = {}
            import copy
            copy_dict = copy.deepcopy(ops)
            for op in ops:
                formatted_json_gold = self.create_dict(op, "gold")
                #print('FJG: ', formatted_json_gold)
                if(event_dict.get("gold_triggers") is None):
                    event_dict["gold_triggers"] = []
                event_dict["gold_triggers"].append(formatted_json_gold.pop("gold_triggers"))
            
            if(event_dict.get("gold_triggers") is None):
                event_dict["gold_triggers"] = {}

            for pop in pops:#copy_dict:
                formatted_json_pred = self.create_dict(pop, "pred")
                if(event_dict.get("predicted_triggers") is None):
                    event_dict["predicted_triggers"] = []
                event_dict["predicted_triggers"].append(formatted_json_pred.pop("predicted_triggers"))

            if(event_dict.get("predicted_triggers") is None):
                event_dict["predicted_triggers"] = {}
            #formatted_json_gold["predicted_triggers"] = formatted_json_pred.pop("predicted_triggers")
            #formatted_json_gold["email_body"] = input_email
            event_dict["email_body"] = re.sub('[ \t]+', ' ', (((email.split('[BOT]')[0]).strip().split('[CONTEXT]')[-1]).replace(',', ' ,').replace('?', ' ?').replace(".", " .").replace("'", " '").strip())).split()
            event_dict["email_thread"] = full_input
            event_dict["full_label"] = full_label
            event_dict["predicted_full_label"] = full_pred_label
            #event_dict["file_name"] = file_name
            #print(json.dumps(event_dict, indent = 4))
            #print('--'*50)
            event_dict["file_name"] = file_name
            self.eval_dict.append(event_dict)
            # print('Output ', self.extract_args_from_template(gt))
            # if(len(op)<=0):
            #     print('*'*100)
            #     print(gt)
            #     print('*'*100)

        loss = outputs[0]
        loss = torch.mean(loss)  
        return loss  
    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            'val/loss': avg_loss, 
        } 
        with open(f'./{self.hparams.op_json_dir}/{self.step_count}.json', 'w') as f:
            json.dump(self.eval_dict, f, indent=4)
        print("dumped to ", f'./{self.hparams.op_json_dir}/{self.step_count}.json')

        metrics = evaluate_old.trigger_scores(self.eval_dict, self.step_count)

        new_thread = metrics.pop("threads")
        with open(f'./{self.hparams.op_json_dir}/{self.step_count}_w_indices.json', 'w') as f:
            json.dump(new_thread, f, indent=4)
        flatten_metric = dict(['Eval_' + parent_metric + child_metric, metrics[parent_metric][child_metric]] for parent_metric in metrics.keys() for child_metric in metrics[parent_metric].keys())
        
        current_metric = flatten_metric["Eval_EM_trigger_id_scoresF1"]
        if(current_metric>self.best_metric):
            self.best_metric = current_metric
            self.best_json_file = self.step_count
        print(f">>>>>>>>> Best JSON file as of now: {self.best_json_file} <<<<<<<<<<")
        for metric in flatten_metric:
            self.log(metric, flatten_metric[metric])
        self.eval_dict = []
        self.step_count += 1

        return {
            'loss': avg_loss, 
            'log': log 
        }   
    def create_dict(self, event_dict, temp_type):
        #print('event_dict: ', event_dict)
        event_type = event_dict.pop('event_type')
        event_trigger = event_dict.pop('event_trigger')
        msr = event_dict.pop("meta_srs")
        my_json = {}
        if(temp_type=="pred"):
            temp_type = "predicted_triggers"
        else:
            temp_type = "gold_triggers"
        my_json[temp_type] = {}
        my_json[temp_type]["arguments"] = []
        if(my_json.get(temp_type) is None):
            my_json[temp_type] = []

        my_json[temp_type]["span"] = re.sub('[ \t]+', ' ', event_trigger.replace(',', ' ,').replace('?', ' ?').replace(".", " .").replace("'", " '").strip())
        my_json[temp_type]["type"] = event_type
        my_json[temp_type]["indices"] = []
        for arg_key, arg_value in event_dict.items():
            arg_key, arg_value = arg_key.strip(), arg_value.strip()
            if(arg_key==arg_value):
                arg_value = ""
                continue#there is no arg in the template, so skip?
            my_json[temp_type]["arguments"].append({"span": re.sub('[ \t]+', ' ', arg_value.replace(',', ' ,').replace('?', ' ?').replace(".", " .").replace("'", " '").strip()), "type":arg_key})
        my_json[temp_type]["arguments"].append({"meta_srs": msr if msr is not None else "Other"})
        return my_json     
    def extract_args_from_template(self, generated_templates):
        ret_outputs = []
        import re
        all_templates = generated_templates.split(self.tokenizer.end_of_template)
        for template in all_templates:
            if(template.strip()==""):
                continue
            template = template.replace(self.tokenizer.begin_of_template, '').strip()
            event_template = r'Event (.+?) is triggered by \| (.+?) \| where , '
            event_trigger = re.match(event_template, template)
            try:
                event, trigger = event_trigger.groups()
                if(event is None):
                    event = ""
                if(trigger is None):
                    trigger = "" 
                dummy_event = template_function_call[event]({}, '', 'trigger')
            except Exception as e:
                continue#if the template is faulty or if there is a mistake in event name/trigger no need to check further
            replace_string = event_template.replace('(.+?)', '{}').replace('\\', '').format(*(event_trigger.groups()))
            template = template.replace(replace_string, '')
            candidate_templates = dummy_event.masked_templates
            candidate_templates = [(re.sub('[ \t]+', ' ', x[0]), x[1]) for x in candidate_templates]
            template = re.sub('[ \t]+', ' ', template)
            for candidate_template in candidate_templates:
                ct = re.sub("[ \t]+", " ", candidate_template[0].replace(",", " ,").replace("?", " ?").replace(".", " .").replace("'", " '"))
                ct = ct.replace('{}', '\| (.+?) \|')
                matches = re.match(r'{}'.format(ct), template)
                if(matches is not None):
                    attrib = candidate_template[1]
                    masked_template = re.sub('[ \t]+', ' ', template_function_call[event]({}, attrib, 'trigger').fill_template().replace(",", " ,").replace("?", " ?").replace(".", " .").replace("'", " '").strip())
                    attrib_names = self.extract_args(masked_template)
                    attrib_valus = self.extract_args(template)
                    display_output = '### '.join([(x + ":" + y) for x, y in zip(attrib_names, attrib_valus)])
                    ret_dict = dict([(x, y) for x, y in zip(attrib_names, attrib_valus)])
                    ret_dict['event_trigger'] = trigger
                    ret_dict['event_type'] = event
                    ret_dict["meta_srs"] = attrib
                    ret_outputs.append(ret_dict)
                    break
        return ret_outputs
        raise NotImplementedError()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        return batch
    def test_epoch_end(self, outputs):
        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name),'w') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):
                    
                    pred = {
                        'doc_key': tup[0][idx],
                        'predicted': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[2][idx].squeeze(0), skip_special_tokens=True) 
                    }
                    writer.write(json.dumps(pred)+'\n')
        return {} 
    def configure_optimizers(self):
        self.train_len = len(self.train_dataloader())
        if self.hparams.max_steps > 0:
            t_total = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        logger.info('{} training steps in total.. '.format(t_total)) 
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler is called only once per epoch by default 
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }
        return [optimizer, ], [scheduler_dict,]
