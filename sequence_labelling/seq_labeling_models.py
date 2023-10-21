import transformers
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertConfig, BertModel
from tqdm import tqdm
from math import log
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

class seq2SeqBERTMC(torch.nn.Module):
    def __init__(self, tokenizer, labels_to_ids, pos_weights):
        super(seq2SeqBERTMC, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained("bert-large-uncased", num_labels=len(labels_to_ids))#<<new
        self.num_labels = len(labels_to_ids)
        if(pos_weights is not None):
            pos_weights = torch.tensor(pos_weights, dtype = torch.float)
        else:
            pos_weights = None
        print('Init weights for loss: ', pos_weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight = pos_weights, ignore_index = -100)
        self.bert.resize_token_embeddings(len(tokenizer))
    def forward(self, input_ids, attention_mask, loss_mask, offset_mapping, trigger_span, labels = None, msr_labels = None, file_name=None):
        logits = self.bert(input_ids = input_ids, attention_mask = attention_mask).logits#<<new
        cls_logits = logits[:, 0, :]#BERT's output dim is [BS, MAX_SEQ_LEN, d], and 0th Seq is CLS (the second dim)
        loss_ = None
        if labels is not None:
            loss_ = 0
            for logit, label, lm in zip(logits, labels, loss_mask):
                logit = logit[lm==1]
                label = label[lm==1]
                loss_ += self.criterion(logit, label)#.mean()
            if(msr_labels is not None):
                loss_ += self.criterion(cls_logits, msr_labels)
                return SequenceClassifierOutput(loss = loss_, logits = {"logits": logits, "msr_logits": cls_logits}, attentions=None)
        return SequenceClassifierOutput(loss = loss_, logits = {"logits": logits, "msr_logits": cls_logits}, attentions=None)