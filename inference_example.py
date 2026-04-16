from system_prompts import TABC_LONG_PROMPT, TABC_PROMPT, GEN_PROMPT, TABC_LONG_ALIGN_PROMPT, TABC_ALIGN_PROMPT
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--method", type=str, default="rlcr",  help="Model name")
args = parser.parse_args()
method = args.method

if method == "rlcr_hotpot":
    model = "gguk2on/qwen2.5-7B-rlcr_g8_b512" 
    prompt_name = "TABC_LONG_PROMPT" 
elif method == "rlar_hotpot":
    model = "gguk2on/qwen2.5-7B-rlar_g8_b512_0.40.15"
    prompt_name = "TABC_LONG_ALIGN_PROMPT" 
elif method == "rlvr_hotpot":
    model = "gguk2on/qwen2.5-7B-rlvr_g8_b512" 
    prompt_name = "GEN_PROMPT" 
elif method == "rlcr_math":
    model = "gguk2on/qwen2.5-7B-rlcr_g32_b384_math"
    prompt_name = "TABC_PROMPT"
elif method == "rlar_math":
    # model = "gguk2on/qwen2.5-7B-rlar_g8_b512_0.40.15" # rollout: 8
    model = "gguk2on/qwen2.5-7B-rlar_g32_b384_math" # rollout: 32
    prompt_name = "TABC_ALIGN_PROMPT"  
elif method == "rlvr_math":
    model = "gguk2on/qwen2.5-7B-rlvr_g8_b384_math"
    prompt_name = "GEN_PROMPT"   


question = "The 2011–12 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was founded in what year?" 

if prompt_name == "TABC_LONG_PROMPT":
    sys_prompt = TABC_LONG_PROMPT
elif prompt_name == "TABC_PROMPT":
    sys_prompt = TABC_PROMPT 
elif prompt_name == "GEN_PROMPT":
    sys_prompt = GEN_PROMPT
elif prompt_name == "TABC_LONG_ALIGN_PROMPT":
    sys_prompt = TABC_LONG_ALIGN_PROMPT
elif prompt_name == "TABC_ALIGN_PROMPT":
    sys_prompt = TABC_ALIGN_PROMPT

user_format = (
                f"\n\nPROBLEM: {question}\n\n"
                )
prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ]

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
to_tokenize = [prompt]
prompt_ids = tokenizer.apply_chat_template(to_tokenize,add_generation_prompt=True)
texts = [tokenizer.decode(x) for x in prompt_ids]

sampling_params= SamplingParams(n = 1, temperature = 0, max_tokens=4096, seed=42) 
llm = LLM(model=model,gpu_memory_utilization=0.9)
outputs = llm.generate(texts,sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
