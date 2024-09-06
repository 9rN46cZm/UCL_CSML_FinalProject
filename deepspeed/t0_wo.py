#!/usr/bin/env python

# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import deepspeed
import os
import torch

model_name = "google-t5/t5-11b"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# initialise Deepspeed ZeRO and store only the engine object
model.eval()  # inference

text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model.to("cuda:0")
inputs = tokenizer.encode(text_in, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    outputs = model.generate(inputs.to("cuda:0"))
text_out = tokenizer.decode(outputs[0].to("cpu"), skip_special_tokens=True)
print(f"in={text_in}\n  out={text_out}")