from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
logger = logging.getLogger(__name__)

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, tp_plan="auto")

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
logger.info(
    f"rank {model.device} ------ {tokenizer.decode(generated[0][inputs['input_ids'].shape[-1] :])}"
    )
print(
    f"rank {model.device} ------ {tokenizer.decode(generated[0][inputs['input_ids'].shape[-1] :])}"
    )
logger.info(
    f"rank {model.device} ------ {tokenizer.decode(generated[0][inputs['input_ids'].shape[-1] :])}"
    )
