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

"Mod", 5, "Sort", 1, 4,5,6,9
#%%
