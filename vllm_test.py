from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/data/ljr/llm_experiments/model/qwen3-1.7B_lifted"
)

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=1024)

# Initialize the vLLM engine
llm = LLM(model="/data/ljr/llm_experiments/model/qwen3-1.7B_lifted")

# Prepare the input to the model
prompt = "Give me a short introduction to large language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Set to False to strictly disable thinking
)

# Generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
