from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch 
import pandas as pd
import json


instruction = "\n\nPROBLEM: {question}\n\nEND OF PROBLEM\n\nMODEL'S RESPONSE: {response}\n\nEND OF RESPONSE\n\n"


IGNORE_INDEX: int = -100


# For cls
class ClassifierDataset(Dataset):
    def __init__(self, q_path, tokenizer, is_train=True,): 
        self.q_path = q_path
        self.data = []
        self.tokenizer = tokenizer
        self.is_train = is_train



        with open(self.q_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.answers = [line["is_correct"] for line in self.data]
        if is_train:
            self.prepare_sft_dataset()
        else:
            self.validation()


    def tokenizing(self, lines):
        encoding = self.tokenizer(
            lines,
            return_tensors = "pt",
            # add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=4096 # ours
        )
        
        return encoding
    
    
    def prepare_sft_dataset(self, verbose=True):
        prompts = [instruction.format(question=item["question"], response=item["response"]) for item in self.data]


        self.labels = [int(item) for item in self.answers]
        self.encoding = self.tokenizing(prompts)
        
        # torch.set_printoptions(profile="full") # for code verifying
        if verbose:
            print(f'sample example: {prompts[0]}')
            print(self.labels[0])

        return None
    
    def validation(self, verbose=False):
        # inputs
        self.inputs = [instruction.format(question=item["question"], response=item["response"]) for item in self.data]
        # labels
        self.labels = [int(item) for item in self.answers]
        
        if verbose:
            print(self.inputs[0])
            print(self.labels[0])
            
        return None
    
    def __getitem__(self, index):
        return {
            "input_ids": self.encoding["input_ids"][index],
            "attention_mask": self.encoding["attention_mask"][index],
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)  
        
