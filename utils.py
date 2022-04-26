import pandas as pd
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import torch

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console


class T5DataSet(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """

    def __init__(
            self, dataframe: pd.DataFrame, tokenizer: T5Tokenizer, source_len: int = 0, target_len: int = 0,
            source_text: str = 'source_text', target_text: str = 'target_text'):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_text,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def display_df(df: pd.DataFrame):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)


def validate(console, tokenizer, model, device, loader):
    """
    Function to evaluate model for predictions
    """
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            target_ids = data["target_ids"].to(device, dtype=torch.long)
            source_ids = data["source_ids"].to(device, dtype=torch.long)
            source_mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                     for g in generated_ids]
            targets = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for t in target_ids]
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actual.extend(targets)
        return predictions, actual
