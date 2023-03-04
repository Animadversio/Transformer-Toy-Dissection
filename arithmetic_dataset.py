
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

token_list = "0123456789=+-*/, "
token_encode = {ch: i for i, ch in enumerate(token_list)}
token_decode = {i: ch for i, ch in enumerate(token_list)}
BOS_id = len(token_list)
EOS_id = len(token_list) + 1
PAD_ID = EOS_id + 1
MASK_LABEL_ID = - 100
token_decode[EOS_id] = "[End]"
token_decode[BOS_id] = "[Beg]"
token_decode[PAD_ID] = ""



def tokenize_str(str):
    return [token_encode[ch] for ch in str]


def detokenize_str(token_list):
    return "".join([token_decode[i] for i in token_list])


def plus_task_generator(A, B):
    """Generate a task for addition.
    """
    task = f"{A} + {B} = "
    task_full = f"{A} + {B} = {A+B}"
    return task, task_full


def minus_task_generator(A, B):
    """Generate a task for subtraction.
    """
    task = f"{A} - {B} = "
    task_full = f"{A} - {B} = {A-B}"
    return task, task_full


def mul_task_generator(A, B):
    """Generate a task for multiplication.
    """
    task = f"{A} * {B} = "
    task_full = f"{A} * {B} = {A*B}"
    return task, task_full


def intdiv_task_generator(A, B):
    """Generate a task for integer division.
    """
    task = f"{A} / {B} = "
    task_full = f"{A} / {B} = {A//B}, {A%B}"
    return task, task_full


task_dict = {
    "+": plus_task_generator,
    "-": minus_task_generator,
    "*": mul_task_generator,
    "/": intdiv_task_generator,
}


def generate_task(task_generator, A, B, add_BOS=True, add_EOS=True):
    """Generate a task.
    """
    if isinstance(task_generator, str):
        task_generator = task_dict[task_generator]
    task, task_full = task_generator(A, B)
    task_tok = tokenize_str(task)
    task_full_tok = tokenize_str(task_full)
    if add_BOS:
        task_tok = [BOS_id] + task_tok
        task_full_tok = [BOS_id] + task_full_tok
    if add_EOS:
        task_full_tok = task_full_tok + [EOS_id]
    return task_tok, task_full_tok

#%%
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
class ArithmeticDataset(Dataset):
    """Generate a dataset for arithmetic tasks.
    """
    def __init__(self, task_generators, A_range, B_range, max_len=64):
        self.task_generators = [task_dict[generator] if isinstance(generator, str) else generator for generator in task_generators]
        self.A_range = A_range
        self.B_range = B_range
        self.max_len = max_len

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        task_generator = np.random.choice(self.task_generators)
        A = np.random.randint(*self.A_range)
        B = np.random.randint(*self.B_range)
        task, task_full = generate_task(task_generator, A, B)
        task_full = torch.tensor(task_full)
        task_full_label = task_full.clone()
        task_full_label[:len(task)] = MASK_LABEL_ID  # mask the task prompt part, not counting into loss
        return task_full, task_full_label


def batch_sampler(data_source, batch_size, ):
    batch = []
    label_batch = []
    for i in range(batch_size):
        data, label = data_source[None]
        batch.append(data)
        label_batch.append(label)
    batch = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    label_batch = pad_sequence(label_batch, batch_first=True, padding_value=MASK_LABEL_ID)
    return batch, label_batch

#
# dataset_pm = ArithmeticDataset(["+", "-"], (0, 100), (0, 100))
# # dataloader = DataLoader(dataset_pm, batch_size=4, shuffle=True)
#
# task_full, task_full_label = batch_sampler(dataset_pm, 50)