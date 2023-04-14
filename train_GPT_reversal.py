import os
from os.path import join
import sys
import random
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from arithmetic_dataset import EOS_id, BOS_id, PAD_ID, tokenize_str, detokenize_str
from arithmetic_dataset import ArithmeticDataset, batch_sampler, token_encode, token_decode
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling
from arithmetic_dataset import generate_task
operator_dict = {"+": lambda x, y: x + y,
                 "-": lambda x, y: x - y,
                 "*": lambda x, y: x * y,
                 "/": lambda x, y: x // y,
                 }
def test_prediction_rev(model, A, B, operator, show=False):
    """
    real_ans, model_anss, moder_ansstrs = test_prediction(model, 120, 51, "*", show=True)
    """
    prompt_ids, full_ids = generate_task(operator, A, B)
    real_answer = operator_dict[operator](A, B)
    prompt_ids = torch.tensor(prompt_ids).unsqueeze(0).cuda()
    answers = model.generate(prompt_ids, max_length=100, do_sample=True,
                    top_k=0, top_p=0.90, num_return_sequences=5,
                    bos_token_id=BOS_id, eos_token_id=EOS_id, pad_token_id=PAD_ID,)
    answer_ints = []
    answer_strs = []
    for answer in answers:
        answer = answer.tolist()
        if not EOS_id in answer:
            answer_str = detokenize_str(answer[1:])
        else:
            answer_str = detokenize_str(answer[1:answer.index(EOS_id)])
        try:
            answer_int = int(answer_str.split(";")[-1])
            answer_ints.append(answer_int)
            if show:
                print(f"{answer_str} ({answer_int})")
        except:
            answer_ints.append(None)
        answer_strs.append(answer_str)
        # print(detokenize_str(answer[prompt_ids.shape[1]:answer.index(EOS_id)]))
    return real_answer, answer_ints, answer_strs


def evaluate_at_exercise_set(model, exercises_set, print_ans=False):
    mean_acc = 0
    summary_str = ""
    for A, B, operator in exercises_set:
        real_ans, model_anss, moder_ansstrs = test_prediction_rev(model, A, B, operator, show=False)
        accuracies = sum([ans == real_ans for ans in model_anss]) / len(model_anss)
        mean_acc += accuracies
        sumstr = f"{A} {operator} {B} = {real_ans}, model ans {model_anss} accuracy: {accuracies:.2f}"
        summary_str += sumstr + "\n"
        if print_ans:
            print(sumstr)
    mean_acc /= len(exercises_set)
    return mean_acc, summary_str


manual_set = [(1, 3, "*"),
              (2, 3, "*"),
              (10, 20, "*"),
              (10, 20, "+"),
              (10, 20, "-"),
              (5, 155, "*"),
              (3, 30, "*"),
              (5, 12, "*"),
              (9, 11, "*"),
              (19, 13, "+"),
              (11, 111, "+"),
              (11, 112, "*"),]
#%%
sys.path.append('/home/binxu/Github/Transformer-Toy-Dissection')
saveroot = r"D:\DL_Projects\Language\ArithmeticGPT"
saveroot = "/home/binxu/DL_Projects/Language/ArithmeticGPT"
os.makedirs(saveroot, exist_ok=True)

# config = GPT2Config(n_embd=128, n_layer=12, n_head=8, n_positions=128, n_ctx=128,
#                     vocab_size=21, bos_token_id=BOS_id, eos_token_id=EOS_id,)
config = GPT2Config(n_embd=128, n_layer=48, n_head=8, n_positions=128, n_ctx=128,
                    vocab_size=21, bos_token_id=BOS_id, eos_token_id=EOS_id,)
model = GPT2LMHeadModel(config)
model.cuda()
optimizer = AdamW(model.parameters(), lr=5e-4)#lr=10e-4)
expdir = join(saveroot, "run_xl_PMM1000_rev_scratch")
os.makedirs(join(expdir, "ckpt"), exist_ok=True)
#%%
dataset_PMM_rev = ArithmeticDataset(["+r", "-r", "*r"], (0, 1000), (0, 1000))
writer = SummaryWriter(expdir)
for epoch in range(0, 300):
    model.train()
    for i in range(200):
        task_fulls, labels = batch_sampler(dataset_PMM_rev, 512)
        out = model(task_fulls.cuda(), labels=labels.cuda())
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch{epoch}-step{i:03d} {loss.item():.5f}")
        writer.add_scalar("loss", loss.item(), epoch * 100 + i)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch * 100 + i)
        writer.add_scalar("epoch", epoch, epoch * 100 + i)

    model.eval()
    mean_acc, sumstr = evaluate_at_exercise_set(model, manual_set, print_ans=True)
    random_set = [(random.randint(1, 1000), random.randint(1, 1000), random.choice(["+", "-", "*"])) for _ in range(25)]
    mean_acc_rand, sumstr_rand = evaluate_at_exercise_set(model, random_set, print_ans=True)
    random_mult_set = [(random.randint(1, 1000), random.randint(1, 1000), random.choice(["*"])) for _ in range(25)]
    mean_acc_randmult, sumstr_randmult = evaluate_at_exercise_set(model, random_mult_set, print_ans=True)
    random_mult10_set = [(random.randint(1, 10), random.randint(1, 10), random.choice(["*"])) for _ in range(25)]
    mean_acc_randmult10, sumstr_randmult10 = evaluate_at_exercise_set(model, random_mult10_set, print_ans=True)
    writer.add_scalar("mean_acc_fixset", mean_acc, (epoch + 1) * 100)
    writer.add_text("fixset_summary", sumstr, (epoch + 1) * 100)
    writer.add_scalar("mean_acc_randset", mean_acc_rand, (epoch + 1) * 100)
    writer.add_text("randset_summary", sumstr_rand, (epoch + 1) * 100)
    writer.add_scalar("mean_acc_randmultset", mean_acc_randmult, (epoch + 1) * 100)
    writer.add_text("randmultset_summary", sumstr_randmult, (epoch + 1) * 100)
    writer.add_scalar("mean_acc_randmult10set", mean_acc_randmult10, (epoch + 1) * 100)
    writer.add_text("randmult10set_summary", sumstr_randmult10, (epoch + 1) * 100)
    torch.save(model.state_dict(), join(expdir, "ckpt", f"model_PMM1000_epoch{epoch}.pt"))


model.save_pretrained(join(saveroot, "gpt2_xl_arithmetic_PMM_rev_scratch"))
#%% # xl version has 48 layers, and use only 512 batch size while others used 1024 batch size
#%%
question = "120 - 51 = "
input_ids = [BOS_id] + tokenize_str(question)
input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()  # Batch size 1
answers = model.generate(input_ids, max_length=100, do_sample=True,
               top_k=0, top_p=0.90, num_return_sequences=5,
               bos_token_id=BOS_id, eos_token_id=EOS_id, pad_token_id=PAD_ID,)
for answer in answers:
    answer = answer.tolist()
    answer_str = detokenize_str(answer[1:answer.index(EOS_id)])
    print(f"{answer_str}")
# answer_strs = [detokenize_str(answer.tolist()[1:-1]) for answer in answers]
# for answer in answer_strs:
#     print(answer)
#%% Model evaluation
EQU_id = token_encode["="]
SPC_id = token_encode[" "]
#%%

#%%
real_ans, model_anss, moder_ansstrs = test_prediction_rev(model, 112, 1000, "*", show=True)
#%%
mean_acc = 0
for A, B, operator in manual_set:
    real_ans, model_anss, moder_ansstrs = test_prediction_rev(model, A, B, operator, show=True)
    accuracies = sum([ans == real_ans for ans in model_anss]) / len(model_anss)
    print(f"{A} {operator} {B} real ans: {real_ans}, model ans {model_anss} accuracy: {accuracies}")
    mean_acc += accuracies
mean_acc /= len(manual_set)
print(f"mean accuracy: {mean_acc}")
#%%
mean_acc, sumstr = evaluate_at_exercise_set(model, manual_set, print_ans=False)
random_set = [(random.randint(1, 1000), random.randint(1, 1000), random.choice(["+", "-", "*"])) for _ in range(20)]
mean_acc_rand, sumstr_rand = evaluate_at_exercise_set(model, random_set, print_ans=False)
