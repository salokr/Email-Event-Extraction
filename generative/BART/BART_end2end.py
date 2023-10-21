import argparse 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 

import torch 
import wandb 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything


import sys
sys.path.append('src/genie')

#Dataloader
from enron_data_module import enronDataModule
from model import GenIEModel 


logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=['gen','constrained-gen']
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['enron']
    )
    parser.add_argument('--tmp_dir', type=str)
    parser.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    parser.add_argument(
        "--op_json_dir",
        default='gen_outputs_Aug28',
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    parser.add_argument(
        "--load_ckpt",
        default=None,
        type=str, 
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
    )
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--mark_trigger', action='store_true')
    parser.add_argument('--sample-gen', action='store_true', help='Do sampling when generation.')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--eval_only", action="store_true",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--gpus", default=1, help='-1 means train on all gpus')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set seed
    seed_everything(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    
    if not args.ckpt_name:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}lr{}_{}'.format(args.model,  args.train_batch_size * args.accumulate_grad_batches, 
                args.learning_rate, time_str)
    #args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    args.ckpt_dir = os.path.join('./checkpoints/'+args.ckpt_name)
    os.makedirs(args.ckpt_dir)


    lr_logger = LearningRateMonitor() 
    tb_logger = TensorBoardLogger('logs/')

    model = GenIEModel(args)
    if args.dataset == 'enron':
        dm = enronDataModule(args)
    else:
        raise NotImplementedError()



    if args.max_steps < 0 :
        args.max_epochs = args.min_epochs = args.num_train_epochs 
    
    
    
    checkpoint_callback = ModelCheckpoint(save_top_k= 6 , monitor="Eval_EM_trigger_id_scoresF1", save_weights_only=True,filename='{epoch}',dirpath=args.ckpt_dir,  mode = "max")
    early_stop_callback = EarlyStopping(monitor="Eval_EM_trigger_id_scoresF1", mode="max", patience = 5,)

    trainer = Trainer(
        logger=tb_logger,
        min_epochs=args.num_train_epochs,
        max_epochs=args.num_train_epochs, 
        gpus=args.gpus, 
        checkpoint_callback=checkpoint_callback, 
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        num_sanity_val_steps=0, 
        val_check_interval=.5, # use float to check every n epochs 
        precision=16 if args.fp16 else 32,
        callbacks = [lr_logger, early_stop_callback]
        ) 

    if args.load_ckpt:
        model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 

    if args.eval_only: 
        dm.setup('test')
        trainer.test(model, datamodule=dm) #also loads training dataloader 
    else:
        dm.setup('fit')
        trainer.fit(model, dm) 
    


if __name__ == "__main__":
    main()


#python train.py --model=gen --ckpt_name=${CKPT_NAME}     --dataset=enron     --train_file=data/wikievents/train.jsonl     --val_file=data/wikievents/dev.jsonl     --test_file=data/wikievents/test.jsonl     --train_batch_size=2     --val_batch_size=4     --learning_rate=3e-5     --accumulate_grad_batches=8     --num_train_epochs=3     --mark_trigger
#history | grep "apt-get" | tail -n 15
#python train.py --model=gen --ckpt_name=${CKPT_NAME}     --dataset=enron     --train_file=data/wikievents/train.jsonl     --val_file=data/wikievents/dev.jsonl     --test_file=data/wikievents/test.jsonl     --train_batch_size=4     --val_batch_size=4     --learning_rate=3e-5     --accumulate_grad_batches=8     --num_train_epochs=100