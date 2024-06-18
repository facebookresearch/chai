# CHAI

_CHAI_ is an inference time pruning method which clusters attention heads that have similar output together with dynamic determination of clusters.
Details can be found in our paper:
[CHAI: Clustered Head Attention for Efficient LLM Inference (Agarwal et al, 2024).](https://arxiv.org/abs/2403.08058)

This repository as intended as a reference implementation for implementing CHAI. To run CHAI please download [LLaMa]((https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1))) models and run inference. 

You can download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5). The repository follows the same code base as LLama-v1.

## Setup
Apply _CHAI_ patch to the Llama model:
```
cd llama
git apply ../llama.patch
cd ..
````

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```

## Inference

```
torchrun --nproc_per_node 1 example_chai.py --ckpt_dir <ModelFolder> --tokenizer_path <tokenizer.model>
```

## Implementation Details

CHAI is implemented primarily in the Forward Function, in model.py for attention.

&nbsp;
## Citation
CHAI is accepted at [ICML'24](https://icml.cc/Conferences/2024/). Please cite as:
``` bibtex
@inproceedings{
agarwal2024chai,
title={{CHAI}: Clustered Head Attention for Efficient {LLM} Inference},
author={Saurabh Agarwal and Bilge Acun and Basil Homer and Mostafa Elhoushi and Yejin Lee and Shivaram Venkataraman and Dimitris Papailiopoulos and Carole-Jean Wu},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=xcDRx8vzCa}
}
```

### License
See the [LICENSE](LICENSE) file.
