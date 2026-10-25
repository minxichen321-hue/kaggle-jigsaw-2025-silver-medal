#!/usr/bin/env bash
# Original offline installation commands; run inside the Kaggle GPU image.
set -euo pipefail

uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'trl==0.21.0' 'optimum==1.27.0' 'auto-gptq==0.7.1' 'bitsandbytes==0.46.1' 'logits-processor-zoo==0.2.1' 'vllm==0.10.0'
uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'deepspeed==0.17.4' -q
uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'triton==3.2.0'
uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'clean-text'
uv pip install --system --no-index -U --no-deps --find-links='/kaggle/input/jigsaw-packages2/whls/' 'peft' 'accelerate' 'datasets'
