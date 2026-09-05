import json
from transformers import AutoTokenizer
from jinja2 import Environment, BaseLoader
SNAP="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
tok=AutoTokenizer.from_pretrained(SNAP)
tmpl=open("/home/ben/task/templates/gemma3_tool_calling.jinja").read()
tok.chat_template=tmpl

# sample xlam-parsed style
msgs=[{"role":"user","content":"Fetch details for 'ethereum'."},
      {"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"web_chain_details","arguments":"{\"chain_slug\": \"ethereum\"}"}}]}]
tools=[{"type":"function","function":{"name":"web_chain_details","description":"Get chain details.","parameters":{"type":"object","properties":{"chain_slug":{"type":"string","description":"slug","default":"ethereum"}},"required":["chain_slug"]}}}]

full=tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
prompt=tok.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True)
print("=====FULL=====")
print(repr(full))
print("=====PROMPT=====")
print(repr(prompt))
assert full.startswith(prompt), "prompt is not a prefix of full!"
completion=full[len(prompt):]
print("=====COMPLETION=====")
print(repr(completion))
print("ends with <end_of_turn>?", completion.rstrip().endswith("<end_of_turn>"))
