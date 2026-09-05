import json, sys
sys.path.insert(0, '.')
from bfcl_evaluation_code import create_tool_info_from_dict
from inspect_ai.model._openai import openai_chat_tool_param
from transformers import AutoTokenizer

SNAP='/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d'
tok=AutoTokenizer.from_pretrained(SNAP)
tok.chat_template=open('templates/gemma3_tool_calling.jinja').read()

# example BFCL-style tool (NOT a test item; generic)
tools_spec=[{
  "name":"calculate_triangle_area",
  "description":"Calculate the area of a triangle given base and height.",
  "parameters":{"type":"dict","properties":{
      "base":{"type":"integer","description":"The base of the triangle."},
      "height":{"type":"integer","description":"The height of the triangle."},
      "unit":{"type":"string","description":"Unit","default":"cm"}
   },"required":["base","height"]}
}]
tinfos=[create_tool_info_from_dict(t) for t in tools_spec]
oai=[openai_chat_tool_param(t) for t in tinfos]
oai=[json.loads(json.dumps(t, default=lambda o: o.__dict__)) for t in oai]
# openai_chat_tool_param returns typed dicts; ensure plain
def to_plain(x):
    if hasattr(x,'model_dump'): return x.model_dump(exclude_none=True)
    return x
oai=[to_plain(openai_chat_tool_param(t)) for t in tinfos]
msgs=[{"role":"user","content":"What is the area of a triangle with base 10 and height 5?"}]
prompt=tok.apply_chat_template(msgs, tools=oai, add_generation_prompt=True, tokenize=False)
print("=====PROMPT START=====")
print(prompt)
print("=====PROMPT END=====")
print("TOOLS JSON PASSED:")
print(json.dumps(oai, indent=2))

print("\n\n===== FULL WITH ASSISTANT TOOLCALL =====")
asst={"role":"assistant","content":"","tool_calls":[
  {"id":"1","type":"function","function":{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}}
]}
full=tok.apply_chat_template(msgs+[asst], tools=oai, add_generation_prompt=False, tokenize=False)
print(repr(full[-300:]))
print("----- readable -----")
print(full)
