from os.path import join
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from arithmetic_dataset import EOS_id, BOS_id, PAD_ID, tokenize_str, detokenize_str
from arithmetic_dataset import ArithmeticDataset, batch_sampler
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling

saveroot = r"D:\DL_Projects\Language\ArithmeticGPT"
config = GPT2Config(n_embd=128, n_layer=12, n_head=8, n_positions=128, n_ctx=128,
                    vocab_size=20, bos_token_id=BOS_id, eos_token_id=EOS_id,)
model = GPT2LMHeadModel(config)

model.cuda()
optimizer = AdamW(model.parameters(), lr=10e-4)
#%%
dataset_PM = ArithmeticDataset(["+", "-"], (0, 1000), (0, 1000))
writer = SummaryWriter(join(saveroot, "run_PM1000"))
for epoch in range(50):
    for i in range(100):
        task_fulls, labels = batch_sampler(dataset_PM, 512)
        out = model(task_fulls.cuda(), labels=labels.cuda())
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch{epoch}-step{i:03d} {loss.item():.5f}")
        writer.add_scalar("loss", loss.item(), epoch * 100 + i)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch * 100 + i)
        writer.add_scalar("epoch", epoch, epoch * 100 + i)
#%
model.save_pretrained(join(saveroot, "gpt2_tiny_arithmetic_PM"))
#%%
question = "120 + 511 = "
input_ids = [BOS_id] + tokenize_str(question)
input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()  # Batch size 1
answers = model.generate(input_ids, max_length=100, do_sample=True,
               top_k=1, top_p=0.90, num_return_sequences=5,
               bos_token_id=BOS_id, eos_token_id=EOS_id, pad_token_id=PAD_ID,)
answer_strs = [detokenize_str(answer.tolist()) for answer in answers]
for answer in answer_strs:
    print(answer)
#%%
