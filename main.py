from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch import cuda
from torch.utils.data import DataLoader

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
from utils import *
from types import SimpleNamespace
import numpy as np
import os


def train(epoch, tokenizer, model, device, loader, optimizer):
    """
    Function to be called for training with the parameters passed from main function
    """
    model.train()
    for _, data in enumerate(loader, 0):
        target_ids = data["target_ids"].to(device, dtype=torch.long)
        target_labels_ids = target_ids[:, :-1].contiguous()
        target_labels = target_ids[:, 1:].clone().detach()
        target_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100
        source_ids = data["source_ids"].to(device, dtype=torch.long)
        source_mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=source_ids,
            attention_mask=source_mask,
            decoder_input_ids=target_labels_ids,
            labels=target_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def T5Trainer(
        dataframe: pd.DataFrame, source_text,
        target_text, params: SimpleNamespace, console, device,
        output_dir: str = "./outputs/"):
    """
    T5 trainer
    :param dataframe:
    :param source_text:
    :param target_text:
    :param params:
    :param output_dir:
    :return:
    """
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(params["SEED"])  # pytorch random seed
    np.random.seed(params["SEED"])  # numpy random seed
    torch.use_deterministic_algorithms(True)

    # logging
    console.log(f"""[Model]: Loading {params["MODEL"]}...\n""")

    # tokenizer for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(params["MODEL"])
    model = T5ForConditionalGeneration.from_pretrained((params["MODEL"])).to(device)

    console.log(f"[DATA]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    train_data = dataframe.sample(frac=params["TRAIN_SIZE"], random_state=params["SEED"])
    val_data = dataframe.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_data.shape}")
    console.print(f"TEST Dataset: {val_data.shape}\n")

    train_dataset = T5DataSet(
        train_data,
        tokenizer,
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    val_dataset = T5DataSet(
        val_data,
        tokenizer,
        params["MAX_SOURCE_TEXT_LENGTH"],
        params["MAX_TARGET_TEXT_LENGTH"],
    )

    train_params = {
        "batch_size": params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 8,
    }

    val_params = {
        "batch_size": params["VAL_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 8,
    }

    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **val_params)

    optimizer = torch.optim.AdamW(

    )

    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(params["EPOCHS"]):
        train(epoch, tokenizer, model, device, train_loader, optimizer)
        console.log(f"[Initiating Validation] on {epoch + 1} EPOCH...\n")
        predictions, actual = validate(tokenizer, model, device, val_loader)
        final_df = pd.DataFrame

    console.log(f"[Saving Model]...\n")
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)





if __name__ == '__main__':
    console = Console(record=True)
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
