import os
from collections import defaultdict
from pathlib import Path

import bitsandbytes as bnb
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from arcot.dataloader import ARCOTDataModule, ARCOTDataset, decoder_collate

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")


arc_fpath = Path("data/original/training")
annotations_train_fpath = Path("data/train.tsv")
annotations_test_fpath = Path("data/test.tsv")
LOG_FREQ = 64
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_RANK = 32
MODEL_NAME = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_NAME = "M4-ai/tau-0.5B"


class PretrainedLMDecoder(L.LightningModule):
    def __init__(self, model, tokenizer, lr=1e-4):
        super().__init__()
        self.model = model
        self.val_step_outs = defaultdict(list)
        self.lr = lr
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        x, lengths, query, answer, filename = batch

        outputs = self.model(**x, labels=x.input_ids)

        if batch_idx % LOG_FREQ == 0:
            with torch.no_grad():
                inputs = self.tokenizer(query[0], return_tensors="pt")
                output = self.model.generate(
                    **inputs, do_sample=True, max_new_tokens=1000
                )[0]
                output = self.tokenizer.decode(output)
                self.logger.experiment.log(
                    {
                        # "train/generate_query": query[0],
                        "train/generate_predicted": output,
                        "train/generate_answer": answer[0],
                        "train/filename": filename[0],
                    }
                )
            # self.logger.experiment.log("train/generate_predicted", output)
            # self.logger.experiment.log("train/generate_answer", answer[0])

        loss = outputs.loss
        self.log("train/loss", loss)
        # acc = (pred_numeric == y).float().mean()
        # self.log("train/acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, lengths, query, answer, filename = batch

        outputs = self.model(**x, labels=x.input_ids)

        # pred_ids = outputs.logits.argmax(-1)
        # pred = [self.tokenizer.decode(x[:l]).strip() for x, l in zip(pred_ids, lengths)]

        if batch_idx % LOG_FREQ == 0:
            with torch.no_grad():
                inputs = self.tokenizer(query[0], return_tensors="pt")
                output = self.model.generate(
                    **inputs, do_sample=True, max_new_tokens=1000
                )[0]
                output = self.tokenizer.decode(output)
                self.logger.experiment.log(
                    {
                        "val/generate_predicted": output,
                        "val/generate_answer": answer[0],
                        "val/filename": filename[0],
                    }
                )
            # self.logger.experiment.log("val/generate_predicted", output)
            # self.logger.experiment.log("val/generate_answer", answer[0])
            # columns = ["input", "prediction"]
            # data = list(zip(text, pred))
            # self.logger.log_text(key="val/preds", columns=columns, data=data)

        loss = outputs.loss
        self.log("val/loss", loss)
        # self.val_step_outs["pred"] += pred_numeric
        # self.val_step_outs["y"] += y
        return loss

    def on_validation_epoch_end(self):
        return
        all_preds = torch.stack(self.val_step_outs["pred"])
        all_labels = torch.stack(self.val_step_outs["y"])
        accuracy = (all_preds == all_labels).float().mean()
        self.log("val/acc", accuracy)
        self.val_step_outs.clear()

    def configure_optimizers(self):
        optimizer = bnb.optim.Adam8bit(self.parameters(), lr=self.lr)
        return optimizer


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, pad_token="</s>"
)
tokenizer.pad_token = tokenizer.eos_token
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_RANK,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, peft_config)
# model.gradient_checkpointing_disable()
model.gradient_checkpointing_enable()


wandb_logger = WandbLogger(project="COLX523")
checkpoint_callback = ModelCheckpoint(
    dirpath="/data3/robinysh/models/colx523",
    filename="{epoch}-{val/acc:.2f}",
    save_top_k=1,
    monitor="val/loss",
    every_n_epochs=1,
    mode="max",
)

dm = ARCOTDataModule(
    arc_fpath=arc_fpath,
    annotations_train_fpath=annotations_train_fpath,
    annotations_test_fpath=annotations_test_fpath,
    tokenizer=tokenizer,
    batch_size=1,
    dataset=ARCOTDataset,
    collate_fn=decoder_collate,
)
model = PretrainedLMDecoder(model=model, tokenizer=tokenizer, lr=1e-5)
# precision = BitsandbytesPrecision(mode="nf4-dq")
# precision = BitsandbytesPrecision(mode="int8-training", dtype=torch.float16, ignore_modules={"lm_head"})
trainer = L.Trainer(
    precision="bf16-true",
    # plugins=precision,
    accelerator="cuda",
    logger=wandb_logger,
    val_check_interval=1.0,
    log_every_n_steps=LOG_FREQ,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=16,
)
trainer.fit(model=model, datamodule=dm)
