import torch
from torch import nn
from arithmetic_dataset import generate_task, \
    plus_task_generator, minus_task_generator, mul_task_generator, intdiv_task_generator

decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=8, dim_feedforward=256, batch_first=True)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6, )
#%%
out = transformer_decoder(torch.rand(10, 10, 64), torch.rand(10, 0, 64))
#%%
from arithmetic_dataset import EOS_id, BOS_id, tokenize_str, detokenize_str
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling

config = GPT2Config(n_embd=128, n_layer=12, n_head=8, n_positions=128, n_ctx=128,
                    vocab_size=20, bos_token_id=BOS_id, eos_token_id=EOS_id,)
#%%
model = GPT2LMHeadModel(config)
#%%
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# GPT2Config.from_pretrained('gpt2')
#%% test
from transformers import Trainer, TrainingArguments
inputs = torch.randint(0, 20, (10, 15))
out = model(inputs, labels=inputs)
# https://github.com/huggingface/transformers/issues/1394#issuecomment-538071860
#%%
from arithmetic_dataset import ArithmeticDataset, batch_sampler, detokenize_str, PAD_ID, EOS_id, BOS_id, tokenize_str
#%%
dataset_PM = ArithmeticDataset(["+", "-"], (0, 100), (0, 100))
for i in range(100):
    task_fulls, labels = batch_sampler(dataset_PM, 10)
    # print(detokenize_str(task))
    print([detokenize_str(task_full.tolist()) for task_full in task_fulls])
#%%
task_prompt, task_full = generate_task("+", 1, 2)
torch.tensor(task_full, dtype=torch.long, )

#%%

