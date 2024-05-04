import json
import random
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from arcot.featurizer import get_objects, get_size
from arcot.prompt import output_format_prompt
from icecream import ic


def decoder_collate(batch, tokenizer):
    query, answer, filename = zip(*batch)
    data = [f"{q}{a}<|endoftext|>" for q, a, *_ in batch]
    inputs = tokenizer.batch_encode_plus(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        # truncation=False,
    )
    encoded_sents = [
        tokenizer.tokenize(sent, padding=True, truncation=True) for sent in data
    ]

    lengths = torch.Tensor([len(data) for data in encoded_sents]).long()
    # ic(lengths)
    pad_id = tokenizer.encode("</s>")[-1]
    inputs['input_ids'][inputs['input_ids'] == pad_id] = -100
    return inputs, lengths, query, answer, filename


class ARCOTDataset(Dataset):
    def __init__(self, arc_fpath, annotations_fpath):
        self.arc = self.load_arc(arc_fpath)
        self.annotations = self.load_annotations(annotations_fpath)

    def load_arc(self, fpath):
        json_files = fpath.glob("*.json")
        data = {}
        for json_fpath in json_files:
            with open(json_fpath, encoding="utf-8") as json_file:
                data[str(json_fpath.stem)] = json.load(json_file)
        return data

    def load_annotations(self, fpath):
        annotations = pd.read_csv(fpath, delimiter="\t")
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        while True:
            annotation = self.annotations.loc[idx]
            filename = annotation.Filename
            annotation = annotation.drop("Filename")
            arc = self.arc[filename.split(".")[0]]

            arc_features = []
            for item in arc["train"]:
                item_features = {}
                for io_type in ["input", "output"]:
                    item_features |= {
                        f"{io_type}_grid_size": get_size(item[io_type]),
                    }
                    obj_repr = get_objects(
                        item[io_type],
                        diag=True,
                        by_color=False,
                        by_row=False,
                        by_col=False,
                        multicolor=False,
                        more_info=True,
                    )
                    pixel_repr = item[io_type]
                    if len(str(obj_repr)) > len(str(pixel_repr)):
                        item_features |= {
                            # f"{io_type}_pixels": item[io_type],
                            f"{io_type}_objects": obj_repr
                        }
                    else:
                        item_features |= {
                            f"{io_type}_pixels": item[io_type],
                        }
                arc_features.append(item_features)
            arc_representation = str(arc_features).replace(", ", ",").replace(" ", "")
            answer = str(annotation.to_dict())
            if len(arc_representation) > 2500:
                idx = random.randint(0, len(self) - 1)
                continue
            # ic(len(str(system_prompt)))
            # ic(len(str(output_format_prompt)))
            # ic(len(str(arc_features)))
            # ic(len(str(annotation)))
            query = ""
            # output += system_prompt
            query += output_format_prompt
            query += arc_representation

            query = f"<|user|>\n{query}<|endoftext|>\n<|assistant|>\n"
            return query, answer, filename


class ARCOTDataModule(L.LightningDataModule):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        arc_fpath,
        annotations_train_fpath,
        annotations_test_fpath,
        batch_size: int = 1,
        tokenizer=None,
        dataset=ARCOTDataset,
        collate_fn=None,
    ):
        super().__init__()
        self.arc_fpath = Path(arc_fpath)
        self.annotations_train_fpath = Path(annotations_train_fpath)
        self.annotations_test_fpath = Path(annotations_test_fpath)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collate_fn = collate_fn

    def train_dataloader(self):
        dataset = self.dataset(self.arc_fpath, self.annotations_train_fpath)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.collate_fn(x, self.tokenizer),
            num_workers=8,
        )

    def val_dataloader(self):
        dataset = self.dataset(self.arc_fpath, self.annotations_test_fpath)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: self.collate_fn(x, self.tokenizer),
            num_workers=8,
        )
