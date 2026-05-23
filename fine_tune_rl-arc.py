from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, LlamaForSequenceClassification, LlamaConfig, T5ForConditionalGeneration
from huggingface_hub import login
import argparse
import torch
from dataset_rl_arc import ClassifierDataset
import utils

# for utilizing GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#### argument #### 
parser = argparse.ArgumentParser()
# required = True
parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=True, default="Qwen/Qwen2.5-7B")
parser.add_argument('--seed', help='seed (2, 10, or 42)', type=int, required=True)

# required = False
parser.add_argument('--per_device_train_batch_size', help='total batch size / # of gradient accumulation steps', type=int, required=False, default=16)
parser.add_argument('--gradient_accumulation_steps', help='# of gradient accumulation steps', type=int, required=False, default=4)
parser.add_argument('--save_path', help='path where the aligner model ckpt to be saved', type=str, required=False, default='./models')
parser.add_argument('--logging_dir', help='path where the logging of aligner model to be saved', type=str, required=False, default="./runs")
parser.add_argument('--lr', help='learning rate', type=float, required=False, default=5e-6)
parser.add_argument('--lr_scheduler_type', help='learning rate scheduler', type=str, required=False, default="linear")
parser.add_argument('--epoch', help='training epoch', type=int, required=False, default=1)
parser.add_argument('--lr_warmup_ratio', help='warmup step ratio, which is # of steps ("total steps * ratio")', type=float, required=False, default=0.05) # llama
parser.add_argument('--weight_decay', help='weight decay', type=float, required=False, default=0.0)
parser.add_argument('--huggingface_api_key', help='huggingface api key for gemma, llama ...', type=str, required=False, default="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")
args = parser.parse_args()


seed = args.seed
model_ckpt = args.model_ckpt
model_name = model_ckpt.split("/")[1]
lr = args.lr
lr_scheduler_type = args.lr_scheduler_type
epoch = args.epoch
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
lr_warmup_ratio = args.lr_warmup_ratio
weight_decay = args.weight_decay


num_labels = 1
question_path = "./dataset/big-math-digit.json"

save_path = f"{args.save_path}/{model_name}"
logging_dir = f"{args.logging_dir}/{model_name}"


##############################################################
################# main code ##################################
##############################################################
api_key = args.huggingface_api_key
login(token=api_key)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# eos랑 pad를 같게 해선 안된다. eos에 대한 로스가 사라져, 문장이 그냥 길어짐.
tokenizer.padding_side = "right" # standard methods

### main model ###f
base_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.pad_token_id = tokenizer.pad_token_id



dataset = ClassifierDataset(question_path, tokenizer, True) # OURS
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
print("### loaded dataset ###")

training_args = TrainingArguments(
    output_dir=save_path,
    logging_strategy='steps',
    logging_steps=50,
    torch_compile=True,
    save_strategy="epoch",
    num_train_epochs=epoch,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=lr_warmup_ratio,
    weight_decay=weight_decay,
    seed=seed,
    report_to='tensorboard',
    logging_dir=logging_dir,
    gradient_checkpointing=False
)

trainer = Trainer(
    model=base_model.to(device),
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print('### start fine-tuning ###')
base_model.config.use_cache=False
trainer.train()
print('### ended fine-tuning ###')
