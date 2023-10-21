import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from seq_labeling_trigger_loader import seq_tagger_trigger_multi_class, seq_tagger_trigger_multi_class_con_sep
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

import wandb, statistics, pprint


run = wandb.init(project="MailEx", entity="salokr", mode="offline")
wandb.run.name="current_turn_only"


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


def prepare_data(MAX_LEN, labels_to_ids, data_address = "./../models2/data"):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    tokenizer.add_tokens(['[TYPE]', '[CONTEXT]'])
    TRAIN_DATA = seq_tagger_trigger_multi_class_con_sep('./data/train', tokenizer, labels_to_ids=labels_to_ids, do_upsampling = False)
    labels_to_ids, ids_to_labels = TRAIN_DATA.get_labels_to_ids()
    VALID_DATA = seq_tagger_trigger_multi_class_con_sep('./data/dev', tokenizer, labels_to_ids = labels_to_ids)
    TEST_DATA = seq_tagger_trigger_multi_class_con_sep('./data/test', tokenizer, labels_to_ids = labels_to_ids)
    return TRAIN_DATA, VALID_DATA, TEST_DATA, tokenizer



dict_count_current = {'Deliver_Data': 18, 'Deliver_Action_Data': 11, 'Request_Action': 8, 'Request_Data': 5, 'Request_Meeting': 5, 'Request_Meeting_Data': 2, 'Deliver_Meeting_Data': 5, 'Amend_Data': 7, 'Request_Action_Data': 2, 'Amend_Meeting_Data': 1}

labels_to_ids = {'O':0}
for label in dict_count_current:
    labels_to_ids['B_' + label] = len(labels_to_ids)
    labels_to_ids['I_' + label] = len(labels_to_ids)
    labels_to_ids['S_' + label] = len(labels_to_ids)

#to check if the model is correctly initialized or not
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
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels, loss_mask = loss_mask, offset_mapping = offset_mapping, trigger_span = trigger_span)
    initial_loss = outputs[0]
    print('initial_loss', initial_loss)
    tr_logits = outputs[1]
    print(len(tr_logits))

import copy
class TriggerOnceTrainer(Trainer):
    def __init__(self, *args, eval_examples = None, post_process_function = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
    def grouped(self, iterable: Iterable[T], n=2) -> Iterable[Tuple[T, ...]]:
        """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), ..."""
        return zip(*[iter(iterable)] * n)
    def createJSONEntry(self, triggers):
        all_triggers = []
        for trigger in triggers:
            trigger_index, trigger_span, trigger_class = [], [], []
            for trg in trigger:
                trigger_index.append(trg[0])
                trigger_class.append(trg[1].replace("B_", "").replace("I_", "").replace("S_", ""))
                trigger_span.append(trg[2])
            all_triggers.append({"span": " ".join(trigger_span), "type": statistics.mode(trigger_class), "indices": trigger_index, "arguments": []})
        return all_triggers
    def dumpObject(self, obj, ff):
        with open(ff, 'wb') as f:
            pickle.dump(obj, f)
            f.close()
    def extract_tag_type_of_events(self, sentence, labels, offset_mapping, event_type, tag_type, file_name):
        labels = [ids_to_labels[x] for x in labels.tolist()]
        triggers, cur_trigger = [], []
        i = 0
        if(tag_type=="S_"):
            while(i<len(offset_mapping)):
                if(labels[i].startswith(tag_type) and labels[i].replace("B_", "").replace("I_", "").replace("S_", "") == event_type):
                    cur_trigger,  shared_triggers, found_I, current_shared_trigger= [(i, labels[i], sentence[i])], [], 0, None
                    i+=1
                    #first extract all the sub tokens
                    while(i<len(offset_mapping) and offset_mapping[i][0]!=0 and offset_mapping[i][1]!=0):
                        cur_trigger.append((i, labels[i], sentence[i]))
                        i+=1
                    shared_len = len(cur_trigger)
                    #now collect all the other parts
                    while(i<len(offset_mapping)):
                        label_event = labels[i].replace("B_", "").replace("I_", "").replace("S_", "")
                        if(label_event==event_type or label_event=="O"):
                            if(labels[i].startswith("S_")):
                                if(found_I>0):
                                    i-=1
                                    break
                                cur_trigger.append((i, labels[i], sentence[i]))
                            elif(labels[i].startswith('O')):
                                if(len(cur_trigger)>(max([len(x) for x in shared_triggers]) if len(shared_triggers)>0 else 1 )and found_I>0 and cur_trigger not in shared_triggers):
                                    #print(cur_trigger, '<<<<<CT', len(shared_triggers))
                                    shared_triggers.append(cur_trigger)
                                    cur_trigger = copy.deepcopy(current_shared_trigger)
                            else: #it must be an I
                                if(current_shared_trigger is None):
                                    current_shared_trigger = copy.deepcopy(cur_trigger)
                                cur_trigger.append((i, labels[i], sentence[i]))
                                found_I += 1
                        i+=1
                    if(len(cur_trigger)>(max([len(x) for x in shared_triggers]) if len(shared_triggers)>0 else 1 )):
                            shared_triggers.append(cur_trigger)
                    if(len(shared_triggers)>0):
                        triggers.extend(shared_triggers)
                else:
                    i+=1
            #print(triggers)
            triggers = [trigger for trigger in triggers if(not all([x[1].startswith("S_") for x in trigger]))]
        elif(tag_type=="B_"):
            while(i<len(offset_mapping)):
                if(labels[i].startswith("B_") and labels[i].replace("B_", "").replace("I_", "").replace("S_", "") == event_type ):
                    cur_trigger = [(i, labels[i], sentence[i])]
                    i+=1
                    while(i<len(offset_mapping) and labels[i].startswith('B') and (offset_mapping[i][0]!=0 and offset_mapping[i][1]!=0)):
                        cur_trigger.append((i, labels[i], sentence[i]))
                        i+=1
                    while(i<len(offset_mapping) and (not labels[i].startswith('S') and not labels[i].startswith('B'))):
                        if(labels[i].startswith('I') and labels[i].replace("B_", "").replace("I_", "").replace("S_", "") == event_type):
                            cur_trigger.append((i, labels[i], sentence[i]))
                        i+=1
                    if(len(cur_trigger)>0):
                        triggers.append(cur_trigger)
                else:
                    i+=1
        else:
            if(len(cur_trigger)>0):
                triggers.append(cur_trigger)
            cur_trigger = []
            i+=1
        return [list(item) for item in set(tuple(row) for row in triggers)] 
    def extract_labels(self, sentence, labels, offset_mapping, file_name):
        triggers = []
        for event_type in dict_count_current.keys():
            for tag_type in ["S_", "B_"]:
                triggers.extend(self.extract_tag_type_of_events(sentence, labels, offset_mapping, event_type, tag_type, file_name))
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
    def extract_batch_labels(self, batch_outputs, loss_masks, batch_labels, inputs, type_, offset_mappings, trigger_spans, file_names):
        batch_predictions, batch_ground_truths, batch_words = [], [], []
        predictions_index, labels_index = [], []
        my_json = []
        macro_f1_id, macro_f1_class, f1_count = 0, 0, 0
        pred_dfs, gold_dfs = [], []
        for email_index, (input_ids, loss_mask, email_label, email_output, offset_mapping, trigger_span, file_name) in enumerate(zip(inputs, loss_masks, batch_labels, batch_outputs, offset_mappings, trigger_spans, file_names)):
            sentence, label, prediction = tokenizer.convert_ids_to_tokens(input_ids[loss_mask==1]), email_label[loss_mask==1], email_output[loss_mask==1]
            #
            x = self.tokenizer.convert_ids_to_tokens(trigger_span)       
            i = (list(g) for _, g in groupby(x, key='[SEP]'.__ne__))
            x = [a + b for a, b in zip(i, i)]
            email_triggers = [' '.join(list(filter(lambda xx: xx not in ['[CLS]', '[SEP]'],xxx))) for xxx in x]
            #
            email_triggers = [x.replace('[CLS]', "").replace("[SEP]", "").strip() for x in email_triggers]
            extracted_triggers = self.extract_labels(sentence, label, offset_mapping[loss_mask==1], file_name)
            gold_triggers_list = self.createJSONEntry(extracted_triggers)
            gold_dfs.extend(gold_triggers_list)
            email_triggers = [x for x in email_triggers if x.strip()!='']
            f1_count+=1
            prediction = prediction.softmax(1).argmax(1)
            predicted_triggers = self.extract_labels(sentence, prediction, offset_mapping[loss_mask==1], file_name)
            predicted_triggers_list = self.createJSONEntry(predicted_triggers)
            pred_dfs.extend(predicted_triggers_list)
            my_json.append({"predicted_triggers": predicted_triggers_list, 'gold_triggers': gold_triggers_list, "email_body": sentence, "full_input": tokenizer.convert_ids_to_tokens(input_ids), "file_name": tokenizer.decode(file_name, skip_special_tokens=True).strip()})
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
        _ = metrics.pop("threads")
        for key in list(metrics.keys()):
            if(not key.startswith("eval_")): 
                metrics["eval_" + key] = metrics.pop(key)
        for key in list(metrics.keys()):
            if(type(metrics[key]) == type({})):
                metric_items = metrics.pop(key)
                for (mk, mv) in metric_items.items():
                    metrics[key + "_" + mk] = mv
        metrics.update({'eval_f_score_id':macro_f1_id/f1_count, 'eval_f_score_class':macro_f1_class/f1_count})
        # Uncomment to see the outputs
        # print('--------------XXX------------------')
        # print('\n'.join([x["span"] for x in gold_dfs]))
        # print('--------------XXX------------------')
        # print('\n'.join([x["span"] for x in pred_dfs]))
        # print('--------------XXX------------------')
        pprint.pprint(metrics)
        return metrics
    def getScores(self, outs, loss_masks, labels, inputs, type, offset_mappings, trigger_spans, file_names):
        outs, loss_masks, labels, inputs, offset_mappings, file_names = torch.tensor(outs), torch.tensor(loss_masks), torch.tensor(labels), torch.tensor(inputs), offset_mappings, torch.tensor(file_names)
        our_score = self.extract_batch_labels(outs, loss_masks, labels, inputs, type, offset_mappings, trigger_spans, file_names)
        return our_score
    def evaluation_loop(self, eval_dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix):
        self.model.eval()
        eval_loss, nb_eval_steps = 0, 0
        complete_outputs, complete_loss_masks, complete_labels, complete_ids, complete_offset_mapping, complete_triggers, complete_file_names = [], [], [], [], [], [], []
        start = time.time()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(eval_dataloader), total = len(eval_dataloader), desc = f'{metric_key_prefix} Metric Calculation'):        
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)
                loss_mask = batch['loss_mask'].to(device, dtype = torch.float)
                offset_mappings = batch["offset_mapping"].to(device)
                trigger_span = batch["trigger_span"]
                file_name = batch["file_name"]
                temp_outputs = self.model(input_ids=ids, attention_mask=mask, labels=labels, loss_mask=loss_mask, offset_mapping = offset_mappings, trigger_span = trigger_span, file_name = file_name)
                complete_outputs.extend(temp_outputs["logits"]["logits"].tolist())
                complete_loss_masks.extend(loss_mask.tolist())
                complete_labels.extend(labels.tolist())
                complete_ids.extend(ids.tolist())
                complete_offset_mapping.extend(offset_mappings)
                complete_triggers.extend(trigger_span)
                complete_file_names.extend(file_name.tolist())
                loss = temp_outputs['loss']
                eval_loss += loss.item()
                nb_eval_steps += 1
            ev_score = self.getScores(complete_outputs, complete_loss_masks, complete_labels, complete_ids, metric_key_prefix, complete_offset_mapping, complete_triggers, complete_file_names)
        ev_score['loss'] = eval_loss/nb_eval_steps
        return ev_score
    def evaluate(self, eval_dataset=None, eval_examples = None, ignore_keys = None, metric_key_prefix = "eval"):
        self._memory_tracker.start()
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        compute_metrics = self.compute_metrics
        start_time = time.time()
        ###        
        start_time = time.time()
        try:
            output_metrics = self.evaluation_loop(eval_dataloader, description = "Evaluation", prediction_loss_only = True if compute_metrics is None else None, ignore_keys = ignore_keys, metric_key_prefix = metric_key_prefix)
        except Exception as e:
            print(e)
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        for key in list(output_metrics.keys()):
            if(not key.startswith("eval_")):
                output_metrics["eval_" + key] = output_metrics.pop(key)
        output_metrics.update({"epoch": self.state.epoch})
        self.log(output_metrics)
        self._memory_tracker.stop_and_update_metrics(output_metrics)
        return output_metrics
    def predict(self, predict_dataset, predict_examples=None, ignore_keys = None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)
        compute_metrics = self.compute_metrics
        start_time = time.time()
        try:
            output_metrics = self.evaluation_loop(predict_dataloader, description = "Prediction", prediction_loss_only=True if compute_metrics is None else None, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            self.compute_metrics = compute_metrics
        for key, value in output_metrics.items():
            if(not key.startswith("pred_")):
                output_metrics["pred_" + key] = output_metrics.pop(key)
        print(json.dumps(output_metrics, indent = 4))
        return output_metrics#PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)

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
    weight_info = TRAIN_DATA.get_labels_frequency_w_inverse_weights()
    labels_frequency, inverse_pos_weights, pos_weights = weight_info["labels_frequency"], weight_info["inverse_pos_weights"], weight_info["weighted_pos"]
    labels_to_ids, ids_to_labels = TRAIN_DATA.get_labels_to_ids()
    torch.manual_seed(1331)
    model = seq2SeqBERTMC(tokenizer, labels_to_ids, None)
    model.to(device)
    assert_model(TRAIN_DATA, model)
    ###
    training_args = TrainingArguments(output_dir="BIOS_class_w_history" , learning_rate=LEARNING_RATE, per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=VALID_BATCH_SIZE, num_train_epochs=EPOCHS, evaluation_strategy="epoch", eval_steps = 1, save_steps = 1,save_strategy="epoch", save_total_limit = 5, load_best_model_at_end=True, push_to_hub=False, metric_for_best_model = 'eval_EM_trigger_class_scores_F1', greater_is_better = True, logging_first_step=True, report_to=['wandb'])
    trainer = TriggerOnceTrainer(model=model, args=training_args, train_dataset=TRAIN_DATA, eval_dataset=VALID_DATA, tokenizer=tokenizer, callbacks = [EarlyStoppingCallback(early_stopping_patience=15, )], post_process_function=post_processing_function,)
    trainer.labels_to_ids=labels_to_ids
    trainer.ids_to_labels=ids_to_labels
    history = trainer.train()
    eval_history = trainer.evaluate()
    return trainer, trainer.state.log_history


TRAIN_DATA, VALID_DATA, TEST_DATA, tokenizer = prepare_data(MAX_LEN, labels_to_ids)
trainer, train_eval_history = train(TRAIN_DATA, VALID_DATA, TEST_DATA, tokenizer, {})
print(train_eval_history)

test_history = trainer.evaluate(TEST_DATA)
print(test_history)

wandb.finish()