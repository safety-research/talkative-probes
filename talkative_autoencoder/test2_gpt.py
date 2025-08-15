from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from transformers.distributed import DistributedConfig

rank=int(os.environ.get("LOCAL_RANK"))

import torch
import torch.distributed as dist
# Initialize the process group
dist.init_process_group("nccl", rank=rank, world_size=int(os.environ.get('WORLD_SIZE')))

model_id = "openai/gpt-oss-20b"
torch.cuda.set_device(rank)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #device_map={"": rank}, #”auto” # either of these two choices leads to the issue!
    tp_plan="auto"
)

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100, do_sample=True)
print(f'rank {rank} ------ {tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:])}')
