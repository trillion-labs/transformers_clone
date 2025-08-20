<!--Copyright 2025 Trillion Labs and the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->


# Tri

## Overview

Tri-70B is the latest and largest flagship language model that redefines the efficiency frontier in LLM training. By achieving frontier performance for it's compute size (1.5T training tokens from scratch), we demonstrate that exceptional capabilities don't require excessive computational resources.

### Key Highlights

- **Architecture optimized for long context**
  - 32k context window
  - Sliding window attention with window size 4096
  - iRoPE: Interleaved local (RoPE) and global (temperature-scaled) attention
  - Scalable softmax
- **Multi-lingual capabilities**: Specially optimized for English, Korean, and Japanese
- **Enhanced reasoning**: Modified training dataset mixture specifically designed for reasoning capabilities, with emphasis on step-by-step problem solving


## Model Details

### Model Specifications

| Specification | Value |
|--------------|-------|
| Type | Causal Language Model |
| Training Stage | Pre-training & Supervised Fine-Tuning |
| Architecture | Transformer Decoder with iRoPE (global attention frequency of 4), SwiGLU, RMSNorm, and GQA |
| Number of Parameters | 70B |
| Number of Layers | 80 |
| Number of Attention Heads | 64 (Query) / 8 (Key, Value) |
| Context Length | 32,768 |
| Number of Tokens Seen | 1.5T |
| Vocab Size | 124,416 |

## Usage examples

For general use, you can use the Tri-70B models with the following example:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "trillionlabs/Tri-70B-preview-SFT"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Explain the concept of central limit theorem in simple terms."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## TriConfig

[[autodoc]] TriConfig

## TriForCausalLM

[[autodoc]] TriForCausalLM

## TriModel

[[autodoc]] TriModel
    - forward

## TriPreTrainedModel

[[autodoc]] TriPreTrainedModel
    - forward