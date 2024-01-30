task_avg_length = {
    "gsm8k" :(76 + 256)/2,
    "gsm8k-cot": (256 + 256)/2,
    "mmlu-cot": (256 + 256)/2,
    "mmlu": (256 + 256)/2,
    "bbh": (256 + 256)/2,
    "bbh-cot": (256 + 256)/2,
    "wikitext": 2048,
}
def uniformQ(q_bit):
    return 16/q_bit
def GroupQ(q_bit):
    return 16/q_bit
def OutlierReducedQ(q_bit,left):
    return 1/(q_bit/16+3*left)
def Gear(q_bit,left,rank,task,head_dim = 32, model_dim = 4096):
    avg_rank = int(task_avg_length[task] * rank)
    total_size = model_dim * head_dim * task_avg_length[task]
    lowrank_size = (task_avg_length[task] + model_dim) * avg_rank/2 # 8bit 
    return total_size/(total_size * (q_bit/16) + 3*left * total_size + lowrank_size)
