# <span style="font-variant: caps;">MailEx: </span>Email Event and Argument Extraction

This repository provides code implementation for our paper [<span style="font-variant: caps;">MailEx: </span>Email Event and Argument Extraction](https://arxiv.org/pdf/2305.08195.pdf) accepted by *EMNLP 2023*.

Please cite our paper if you find our work/code helpful!
```
@misc{srivastava2023mailex,
      title={MAILEX: Email Event and Argument Extraction}, 
      author={Saurabh Srivastava and Gaurav Singh and Shou Matsumoto and Ali Raz and Paulo Costa and Joshua Poore and Ziyu Yao},
      year={2023},
      eprint={2305.13469},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 1. Overview
<p align="center">
<img src="overview.png" alt="Arch Overview" title="Overview" width="600"/>
</p>
In this work, we present the first dataset, MailEx, for performing event extraction from conversational email threads. To this end, we first proposed a new taxonomy covering 10 event types and 76 arguments in the email domain. Our final dataset includes ∼4K emails annotated with ∼9K event instances. To understand the task challenges, we conducted a series of experiments comparing two commonly-seen lines of approaches for event extraction, i.e., sequence labeling and generative end-to-end extraction (including few-shot GPT-3.5). Our results showed that the task of email event extraction is far from being addressed, due to challenges lying in, e.g., extracting non-continuous, shared trigger spans, extracting non-named entity arguments, and modeling the email conversational history. Our work thus suggests more investigations in this domain-specific event extraction task in the future.

## 2. Setup
This project is tested in Python 3.8.6

To get started, first, clone the repository:
```
git clone https://github.com/salokr/Email-Event-Extraction.git
cd Email-Event-Extraction
export MAILEX_HOME=$(pwd)
export PYTHONPATH=$MAILEX_HOME:$MAILEX_HOME/utils:$PYTHONPATH
```

Then download the dataset from [this link](https://drive.google.com/file/d/1a336g4-wlEwsVbXLPB9wPQnBDRE933mb/view?usp=sharing) and save it under the `data/` folder. Your directory structure should like the following:
```
Email-Event-Extraction
│
├── sequence_labelling
│   ├── BERT_argument_only.py
│   ├── BERT_trigger_only.py
│   └── seq_labeling_models.py
│
├── utils
│   ├── seq_labeling_argument_loader.py
│   ├── evaluate.py
│   └── ...
│
├── data
│   ├── train
│   │   └── ...
│   ├── dev
│   │   └── ...
│   └── test
│       └── ...
│
└── generative
    ├── BART
    │   └── ...
    └── ICL
        └── ...
```

Create a virtual environment and install all dependencies:
```
python -m venv mailexenv 
source mailexenv/bin/activate
pip install -r requirements.txt
```

## 2.1 Experiments with Sequence Labeling Models
To experiment with the BERT-based sequence labeling architecture. We first begin with extracting the trigger extraction. To perfrom trigger extraction, please run the following code:

```
python sequence_labelling/BERT_trigger_only.py
```

Then, we feed the extracted triggers to extract the corresponding arguments. Run the following commands to extract arguments :
```
python sequence_labelling/BERT_seq_labeling.py
```
For extracting arguments only with the ground-truth triggers. Run the following:
```
python sequence_labelling/BERT_argument_only.py 
```
## 2.2 Experiments with Generative Models
### 2.2.1 Experiments with BART-based model
For experimenting with generative models, we need to create templates. The preprocessing function in the `gen_data_loader.py` takes care of generating the templates. To perform end-to-end event and argument extraction, run the following:
```
python sequence_labelling/BART_end2end.py
```
### 2.2.2 Experiments with In-context learning
First, generate the templates using the following:
```
python utils/gen_data_loader.py
```
which will create three files namely `train_prompt_data.json`, `dev_prompt_data.josn`, and `test_prompt_data.json`. For the ICL experiments, we need only the test data. Please put the files in `data/icl_data/`.

Run the following command, to perform the experiment.

```
python generative/ICL/prompt_experiments.py --model davinci
```

Change the model flag to `turbo` to experiment with GPT-3.5-turbo.
