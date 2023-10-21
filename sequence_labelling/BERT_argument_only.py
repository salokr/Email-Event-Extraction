import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from seq_labeling_argument_loader import seq_tagger_arg_multi_class
from transformers import AutoTokenizer
from torch import cuda
from sklearn.metrics import classification_report, f1_score
import pickle
from tqdm import tqdm
import json
import os, time
from sequence_labelling.seq_labeling_models import seq2SeqBERTMC
from transformers import TrainingArguments, EarlyStoppingCallback
from transformers import Trainer
import evaluate, math
from itertools import groupby, chain
from typing import Iterable, Tuple, TypeVar

import wandb, statistics


run = wandb.init(project="MailEx", entity="salokr", mode="offline")
wandb.run.name="Argument_Only"


T = TypeVar("T")
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10


wandb.define_metric('eval_trigger_id_scores_Precision', summary = 'max')
wandb.define_metric('eval_trigger_id_scores_Recall', summary = 'max')
wandb.define_metric('eval_trigger_id_scores_F1', summary = 'max')
wandb.define_metric('eval_trigger_class_scores_Precision', summary = 'max')
wandb.define_metric('eval_trigger_class_scores_Recall', summary = 'max')
wandb.define_metric('eval_trigger_class_scores_F1', summary = 'max')
wandb.define_metric('eval_arg_id_scores_Precision', summary = 'max')
wandb.define_metric('eval_arg_id_scores_Recall', summary = 'max')
wandb.define_metric('eval_arg_id_scores_F1', summary = 'max')
wandb.define_metric('eval_arg_class_scores_Precision', summary = 'max')
wandb.define_metric('eval_arg_class_scores_Recall', summary = 'max')
wandb.define_metric('eval_arg_class_scores_F1', summary = 'max')
wandb.define_metric('eval_f_score_id', summary = 'max')
wandb.define_metric('eval_f_score_class', summary = 'max')
wandb.define_metric('eval_loss', summary = 'min')


def prepare_data(MAX_LEN, data_address = "./../models2/data"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(['[TYPE]', '[TRG]', '[/TRG]'])
    TRAIN_DATA = seq_tagger_arg_multi_class('./data/train', tokenizer)
    labels_to_ids, ids_to_labels = TRAIN_DATA.get_labels_to_ids()
    VALID_DATA = seq_tagger_arg_multi_class('./data/dev', tokenizer, labels_to_ids = labels_to_ids)
    TEST_DATA = seq_tagger_arg_multi_class('./data/test', tokenizer, labels_to_ids = labels_to_ids)
    #labels_to_ids, ids_to_labels = TRAIN_DATA.get_labels_to_ids()
    return TRAIN_DATA, VALID_DATA, TEST_DATA, tokenizer

def assert_model(TRAIN_DATA, model):
    inputs = TRAIN_DATA[2]
    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)
    labels = inputs["labels"].unsqueeze(0)
    loss_mask = inputs["loss_mask"].unsqueeze(0)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    loss_mask = loss_mask.to(device)
    offset_mapping  = inputs["offset_mapping"].unsqueeze(0).to(device)
    trigger_span = inputs["trigger_span"]
    msr_labels = inputs["msr_labels"].unsqueeze(0).to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_mask = loss_mask, offset_mapping = offset_mapping, trigger_span = trigger_span, msr_labels = msr_labels)
    initial_loss = outputs[0]
    print('initial_loss', initial_loss)
    tr_logits = outputs[1]["logits"]
    print(tr_logits.shape)


def extract_trigger_and_event_type_from_label(email):
    i2l = TEST_DATA.ids_to_labels
    email = TEST_DATA.tokenizer.convert_ids_to_tokens(email)
    try:
        trigger_start, trigger_end = email.index('[TRG]'), email.index('[/TRG]')
        event_trigger = email[trigger_start+1 : trigger_end]
        trigger_indices = [i for i in range(trigger_start+1 , trigger_end)]
    except:
        event_trigger = ""#the trigger, unfortunately, was trimmed
        trigger_indices = [-1,-1]
    event_type = ' '.join(email[1:email.index('[TYPE]')]).title()
    return {"event_type": event_type, "event_trigger":' '.join(event_trigger), "trigger_indices": trigger_indices}



import copy
class TriggerOnceTrainer(Trainer):
    def __init__(self, *args, eval_examples = None, post_process_function = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
    def grouped(self, iterable: Iterable[T], n=2) -> Iterable[Tuple[T, ...]]:
        """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), ..."""
        return zip(*[iter(iterable)] * n)
    def createJSONEntry(self, triggers, meta_srs, trig_event):
        all_args = []
        trigger_data = {"span": trig_event["event_trigger"], "indices": trig_event["trigger_indices"], "type": trig_event["event_type"]}
        for trigger in triggers:
            trigger_index, trigger_span, trigger_class = [], [], []
            for trg in trigger:
                trigger_index.append(trg[0])
                trigger_class.append(trg[1].replace("B-", "").replace("I-", "").replace("S-", ""))
                trigger_span.append(trg[2])
            all_args.append({"span": " ".join(trigger_span), "type": statistics.mode(trigger_class), "indices": trigger_index})
        trigger_data.update({"arguments":all_args})
        trigger_data["arguments"].append({"meta_srs":meta_srs["meta_span"]})
        return [trigger_data]
    def dumpObject(self, obj, ff):
        with open(ff, 'wb') as f:
            pickle.dump(obj, f)
            f.close()
    def extract_labels(self, sentence, labels, offset_mapping):
        i=0
        triggers, cur_trigger, labels = [], [], [self.ids_to_labels[x.tolist()] for x in labels]
        while(i<len(offset_mapping)):
            if(labels[i].find('S-')>=0):
                cur_trigger,  shared_triggers= [(i, labels[i], sentence[i])], []
                i+=1
                while(i<len(offset_mapping) and labels[i].find('S-')>=0): #and (offset_mapping[i][0]!=0 and offset_mapping[i][1]!=0)):
                    cur_trigger.append((i, labels[i], sentence[i]))
                    i += 1#This step will ensure that all the tokens of this word are covered
                shared_trigger = copy.deepcopy(cur_trigger)
                found_I = 0
                while(i<len(offset_mapping) and (not labels[i].find('S-')>=0 and not labels[i].find('B-')>=0)):
                    if(labels[i].startswith('O')):
                        if(len(cur_trigger)>1 and found_I>0):
                            shared_triggers.append(cur_trigger)
                            cur_trigger = copy.deepcopy(shared_trigger)
                    else: #it must be an I
                        cur_trigger.append((i, labels[i], sentence[i]))
                        found_I += 1
                    i+=1
                if(len(cur_trigger)>1):
                    shared_triggers.append(cur_trigger)
                if(len(shared_triggers)>0):
                    triggers.extend(shared_triggers)
            elif(labels[i].find('B-')>=0):#first collect all Bs
                cur_trigger = [(i, labels[i], sentence[i])]
                i+=1
                while(i<len(offset_mapping) and labels[i].find('B-')>=0 and (offset_mapping[i][0]!=0 and offset_mapping[i][1]!=0)):
                    cur_trigger.append((i, labels[i], sentence[i]))
                    i+=1
                while(i<len(offset_mapping) and (not labels[i].find('S-')>=0 and not labels[i].find('B-')>=0)):
                    if(labels[i].find('I-')>=0):
                        cur_trigger.append((i, labels[i], sentence[i]))
                    i+=1
                if(len(cur_trigger)>0):
                    triggers.append(cur_trigger)
            elif(labels[i].find('I-')>=0):
                while(i<len(offset_mapping) and labels[i].find('I-')>=0 and (offset_mapping[i][0]!=0 and offset_mapping[i][1]!=0)):
                    cur_trigger.append((i, labels[i], sentence[i]))
                    i+=1
                while(i<len(offset_mapping) and (not labels[i].find('S-')>=0 and not labels[i].find('B-')>=0)):
                    if(labels[i].find('I-')>=0):
                        cur_trigger.append((i, labels[i], sentence[i]))
                    i+=1
                if(len(cur_trigger)>0):
                    triggers.append(cur_trigger)
            else:
                if(len(cur_trigger)>0):
                    triggers.append(cur_trigger)
                cur_trigger = []
                i+=1        
        return [list(item) for item in set(tuple(row) for row in triggers)]
    def get_scores(self, extracted_triggers, predicted_triggers):
        flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
        extracted_triggers = list(flatten(extracted_triggers))
        predicted_triggers = list(flatten(predicted_triggers))
        gold_trigger_with_types = list(self.grouped(extracted_triggers))
        pred_trigger_with_types = list(self.grouped(predicted_triggers))
        #
        extracted_triggers = set([et[0] for et in gold_trigger_with_types])
        predicted_triggers = set([pd[0] for pd in pred_trigger_with_types])
        numerator = len(predicted_triggers.intersection(extracted_triggers))
        precision = numerator/len(predicted_triggers) if len(predicted_triggers)>0 else 0
        recall = numerator/len(extracted_triggers) if len(extracted_triggers)>0 else 0
        gold_trigger_with_types_unique = set(gold_trigger_with_types)
        pred_trigger_with_types_unique = set(pred_trigger_with_types)
        class_numerator = len(gold_trigger_with_types_unique.intersection(pred_trigger_with_types_unique))
        class_precision = class_numerator/len(pred_trigger_with_types_unique) if len(pred_trigger_with_types_unique)>0 else 0
        class_recall = class_numerator/len(gold_trigger_with_types_unique) if len(gold_trigger_with_types_unique)>0 else 0
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0
        class_f1 = (2*class_precision*class_recall)/(class_precision + class_recall) if (class_precision + class_recall)>0 else 0
        #
        score={'ID_precision_overlap': precision, 'ID_recall_overlap': recall, 'ID_f_score_overlap':f1}
        score.update({'class_precision_overlap': class_precision, 'class_recall_overlap': class_recall, 'class_f_score_overlap':class_f1})
        return score
    def extract_batch_labels(self, batch_outputs, loss_masks, batch_labels, inputs, type_, offset_mappings, trigger_spans, complete_msr_logits, complete_msr_labels, file_names):
        batch_predictions, batch_ground_truths, batch_words = [], [], []
        predictions_index, labels_index = [], []
        my_json = []
        macro_f1_id, macro_f1_class, f1_count = 0, 0, 0
        pred_dfs, gold_dfs = [], []
        for email_index, (input_ids, loss_mask, email_label, email_output, offset_mapping, trigger_span, msr_logit, msr_label, file_name) in enumerate(zip(inputs, loss_masks, batch_labels, batch_outputs, offset_mappings, trigger_spans, complete_msr_logits, complete_msr_labels, file_names)):
            #print(msr_logit.shape)
            ##
            msr_softmax = torch.softmax(msr_logit, dim = -1)
            msr_output = torch.argmax(msr_softmax)
            msr_output_label = self.ids_to_labels[msr_output.tolist()]
            ##
            sentence, label, prediction = tokenizer.convert_ids_to_tokens(input_ids[loss_mask==1]), email_label[loss_mask==1], email_output[loss_mask==1]
            #
            x = self.tokenizer.convert_ids_to_tokens(trigger_span)       
            i = (list(g) for _, g in groupby(x, key='[SEP]'.__ne__))
            x = [a + b for a, b in zip(i, i)]
            email_triggers = [' '.join(list(filter(lambda xx: xx not in ['[CLS]', '[SEP]'],xxx))) for xxx in x]
            #
            email_triggers = [x.replace('[CLS]', "").replace("[SEP]", "").strip() for x in email_triggers]
            extracted_triggers = self.extract_labels(sentence, label, offset_mapping[loss_mask==1])
            #print(extracted_triggers)
            trig_event = extract_trigger_and_event_type_from_label(input_ids)
            gold_triggers_list = self.createJSONEntry(extracted_triggers, {"meta_index": msr_label.tolist(), "meta_span": self.ids_to_labels[msr_label.tolist()]}, trig_event)
            email_triggers = [x for x in email_triggers if x.strip()!='']
            print_triggers = []
            for x in extracted_triggers:
                if(type(x)==type([])):
                    #print(x)
                    preds = [sentence[ii[0]] if type(ii)==type((0, 1)) else [sentence[iii[0]] for iii in ii] for ii in x]
                    print_triggers.append(preds)
                else:
                    print_triggers.append(sentence[x[0]])
            f1_count+=1
            prediction = prediction.softmax(1).argmax(1)
            print_triggers = []
            predicted_triggers = self.extract_labels(sentence, prediction, offset_mapping[loss_mask==1])
            predicted_triggers_list = self.createJSONEntry(predicted_triggers, {"meta_index": msr_output.tolist(), "meta_span": msr_output_label}, trig_event)
            my_json.append({"predicted_triggers": predicted_triggers_list, 'gold_triggers': gold_triggers_list, "email_body": sentence, "file_name": tokenizer.decode(file_name, skip_special_tokens=True).strip()})
            for x in predicted_triggers:
                if(type(x)==type([])):
                    #print(x)
                    preds = [sentence[ii[0]] if type(ii)==type((0, 1)) else [sentence[iii[0]] for iii in ii] for ii in x]
                    print_triggers.append(preds)
                else:
                    print_triggers.append(sentence[x[0]])
            all_scores = self.get_scores(extracted_triggers, predicted_triggers)
            macro_f1_id += all_scores['ID_f_score_overlap']
            macro_f1_class +=all_scores['class_f_score_overlap']
            batch_predictions.extend(prediction.tolist())
            batch_ground_truths.extend(label.tolist())
            pred_span, gold_span= {}, {}
        epoch = math.ceil(self.state.epoch)
        self.dumpObject([batch_ground_truths, batch_predictions], f"./{self.args.output_dir}/{type_}_span_ID_epoch_{epoch}.res")
        with open(f"./{self.args.output_dir}/{type_}_JSON_epoch_{epoch}.json", "w") as f:
            json.dump(my_json, f, indent = 4)
        metrics = evaluate.trigger_scores(my_json)
        for key in list(metrics.keys()):
            if(not key.startswith("eval_")): 
                metrics["eval_" + key] = metrics.pop(key)
        for key in list(metrics.keys()):
            if(type(metrics[key]) == type({})):
                metric_items = metrics.pop(key)
                for (mk, mv) in metric_items.items():
                    metrics[key + "_" + mk] = mv
        metrics.update({'eval_f_score_id':macro_f1_id/f1_count, 'eval_f_score_class':macro_f1_class/f1_count})
        #
        # print('--------------XXX------------------')
        # print(' '.join([x["span"] for x in gold_dfs]))
        # print('*'*1000)
        # print('\n'.join([x["span"] for x in pred_dfs]))
        # print('--------------XXX------------------')
        #
        # pred_table = wandb.Table(dataframe = pd.DataFrame(pred_dfs))
        # gold_table = wandb.Table(dataframe = pd.DataFrame(gold_dfs))
        # run.log({"Predicted Spans": pred_table})
        # run.log({"Gold Spans": gold_table})
        return metrics
    def getScores(self, outs, loss_masks, labels, inputs, type, offset_mappings, trigger_spans, complete_msr_logits, complete_msr_labels, complete_file_names):
        outs, loss_masks, labels, inputs, offset_mappings, complete_msr_logits, complete_msr_labels, complete_file_names = torch.tensor(outs), torch.tensor(loss_masks), torch.tensor(labels), torch.tensor(inputs), offset_mappings, torch.tensor(complete_msr_logits), torch.tensor(complete_msr_labels), torch.tensor(complete_file_names)
        our_score = self.extract_batch_labels(outs, loss_masks, labels, inputs, type, offset_mappings, trigger_spans, complete_msr_logits, complete_msr_labels, complete_file_names)
        return our_score
    def evaluation_loop(self, eval_dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix):
        self.model.eval()
        eval_loss, nb_eval_steps = 0, 0
        complete_outputs, complete_loss_masks, complete_labels, complete_ids, complete_offset_mapping, complete_triggers, complete_msr_logits, complete_msr_labels, complete_file_names = [], [], [], [], [], [], [], [], []
        start = time.time()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(eval_dataloader), total = len(eval_dataloader), desc = f'{metric_key_prefix} Metric Calculation'):        
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)
                loss_mask = batch['loss_mask'].to(device, dtype = torch.float)
                offset_mappings = batch["offset_mapping"].to(device)
                trigger_span = batch["trigger_span"]
                msr_labels = batch["msr_labels"].to(device)
                temp_outputs = self.model(input_ids=ids, attention_mask=mask, labels=labels, loss_mask=loss_mask, offset_mapping = offset_mappings, trigger_span = trigger_span, msr_labels = msr_labels)
                complete_outputs.extend(temp_outputs["logits"]["logits"].tolist())
                complete_msr_logits.extend(temp_outputs["logits"]["msr_logits"].tolist())
                complete_msr_labels.extend(msr_labels.tolist())
                complete_loss_masks.extend(loss_mask.tolist())
                complete_labels.extend(labels.tolist())
                complete_ids.extend(ids.tolist())
                complete_offset_mapping.extend(offset_mappings)
                complete_file_names.extend(batch["file_name"].tolist())
                complete_triggers.extend(trigger_span)
                loss = temp_outputs['loss']
                eval_loss += loss.item()
                nb_eval_steps += 1
            ev_score = self.getScores(complete_outputs, complete_loss_masks, complete_labels, complete_ids, metric_key_prefix, complete_offset_mapping, complete_triggers, complete_msr_logits, complete_msr_labels, complete_file_names)
        ev_score['loss'] = eval_loss/nb_eval_steps
        return ev_score
    def evaluate(self, eval_dataset=None, eval_examples = None, ignore_keys = None, metric_key_prefix = "eval"):
        self._memory_tracker.start()
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        compute_metrics = self.compute_metrics
        start_time = time.time()
        start_time = time.time()
        try:
            output_metrics = self.evaluation_loop(eval_dataloader, description = "Evaluation", prediction_loss_only = True if compute_metrics is None else None, ignore_keys = ignore_keys, metric_key_prefix = metric_key_prefix)
        except Exception as e:
            print('>>>', e)
            import traceback
            traceback.print_exc()
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        for key in list(output_metrics.keys()):
            if(not key.startswith("eval_")):
                output_metrics["eval_" + key] = output_metrics.pop(key)
        output_metrics.update({"epoch": self.state.epoch})
        self.log(output_metrics)
        self._memory_tracker.stop_and_update_metrics(output_metrics)
        # print(json.dumps(output_metrics, indent = 4))
        return output_metrics
    def predict(self, predict_dataset, predict_examples=None, ignore_keys = None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)
        complete_outputs, complete_msr_logits = [], []
        for idx, batch in tqdm(enumerate(predict_dataloader), total = len(predict_dataloader), desc = "testing"):
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            temp_outputs = self.model(input_ids=ids, attention_mask=mask, labels=None, loss_mask=[], offset_mapping = [], trigger_span = [], msr_labels = None)
            
            msr_softmax = torch.softmax(temp_outputs["logits"]["msr_logits"], dim = -1)
            # print(msr_softmax)
            # print(msr_softmax.shape)
            msr_output = torch.argmax(msr_softmax)
            msr_output_label = self.ids_to_labels[msr_output.tolist()]

            prediction = temp_outputs["logits"]["logits"].softmax(1).argmax(1)

            complete_outputs.extend(prediction.tolist())
            complete_msr_logits.extend(msr_output_label)
        return complete_outputs


def post_processing_function(examples, features, outputs, stage = "eval"):
    preds = outputs.predictions
    if(isinstance(preds, tuple)):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
    example_id_to_index = {k:i for i, k in enumerate(examples["id"])}
    feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
    predictions = {}
    for example_index, example in enumerate(examples):
        feature_index = feature_per_example[example_index]
        predictions[example["id"]] = decoded_preds[feature_index]
    formatted_predictions = [{"id":k, "prediction_text":v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answer_column"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)



def train(TRAIN_DATA, VALID_DATA, TEST_DATA, tokenizer, args):
    global labels_to_ids, ids_to_labels, pos_weights
    train_params = {'batch_size': 4, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': 4,'shuffle': True,'num_workers': 0}
    weight_info = TRAIN_DATA.get_labels_frequency_w_inverse_weights()
    labels_frequency, inverse_pos_weights, pos_weights = weight_info["labels_frequency"], weight_info["inverse_pos_weights"], weight_info["weighted_pos"]
    labels_to_ids, ids_to_labels = TEST_DATA.get_labels_to_ids()
    assert(len(labels_to_ids) == len(ids_to_labels))
    torch.manual_seed(1331)
    model = seq2SeqBERTMC(tokenizer, labels_to_ids, None)
    model.to(device)
    assert_model(TRAIN_DATA, model)
    ###
    training_args = TrainingArguments(output_dir="args_test_new" , learning_rate=LEARNING_RATE, per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=VALID_BATCH_SIZE, num_train_epochs=EPOCHS, evaluation_strategy="epoch", eval_steps = 1, save_steps = 1,save_strategy="epoch", save_total_limit = 5, load_best_model_at_end=True, push_to_hub=False, metric_for_best_model = 'eval_f_score_class', greater_is_better = True, logging_first_step=True, report_to=['wandb'])
    trainer = TriggerOnceTrainer(model=model, args=training_args, train_dataset=TRAIN_DATA, eval_dataset=VALID_DATA, tokenizer=tokenizer, callbacks = [EarlyStoppingCallback(early_stopping_patience=15, )], post_process_function=post_processing_function,)
    trainer.labels_to_ids=labels_to_ids
    trainer.ids_to_labels=ids_to_labels
    history = trainer.train()
    #print(history)
    eval_history = trainer.evaluate()
    return trainer.state.log_history, trainer


TRAIN_DATA, VALID_DATA, TEST_DATA, tokenizer = prepare_data(MAX_LEN)

for t in TRAIN_DATA:
    pass

for t in VALID_DATA:
    pass

for t in TEST_DATA:
    pass

# print((TEST_DATA.labels_to_ids))
train_eval_history, trainer = train(TRAIN_DATA, VALID_DATA, TEST_DATA, tokenizer, {})
print(train_eval_history)
wandb.finish()
trainer.evaluate(TEST_DATA)


###############################################