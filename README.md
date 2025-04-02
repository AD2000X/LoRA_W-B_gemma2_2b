# Gemma 2 Fine-tuning with LoRA and Weights & Biases

This project demonstrates how to efficiently fine-tune the Gemma 2 (2B) language model using Low-Rank Adaptation (LoRA) technique, with experiment tracking via the Weights & Biases platform.
Report link: https://wandb.ai/ad2000x-none/LoRA_gemma_kerasNLP_W&B/reports/Exploring-Project-Weights-Biases-for-Fine-Tuning-a-Large-Language-Model-with-LoRA--VmlldzoxMTQ1MTAwMA?accessToken=ac7h6wqw9m0tjk53954o2e0dlsrpt1otjsaqvhq4rlu60ct1vsg2oknrzh8sbmva

## Features

- Parameter-efficient fine-tuning using LoRA
- Support for different precision training (FP32, mixed_bfloat16, mixed_float16)
- Integration with Weights & Biases for experiment tracking
- Uses Keras and KerasNLP libraries
- Complete evaluation metrics

## Requirements

- Python 3.8+
- CUDA-enabled GPU
- [Weights & Biases](https://wandb.ai/) account

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemma2-lora-finetune.git
cd gemma2-lora-finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

This project uses the Databricks Dolly 15k dataset for instruction fine-tuning. The dataset will be automatically downloaded during training.

### Model Training

1. Set up your Weights & Biases API key:
```python
import wandb
wandb.login()
```

2. Choose a training precision mode (run the corresponding notebook in the notebooks directory):
   - `lora_finetune_fp32.ipynb`: Train with FP32 precision
   - `lora_finetune_bfloat16.ipynb`: Train with mixed_bfloat16 precision
   - `lora_finetune_float16.ipynb`: Train with mixed_float16 precision

### Model Evaluation

After training, you can evaluate the model performance using the evaluation script:
```bash
python src/evaluate.py --model-path path/to/saved/model
```

## Performance Comparison

| Precision Type | Training Time (s/epoch) | Perplexity | Readability (Flesch Score) |
|----------|-------------|------------|------------------------|
| Original Model | - | 1.3992/2.0194 | 64.2/53.58 |
| FP32 | ~45s | 2.9536/1.6566 | 51.52/28.17 |
| mixed_bfloat16 | ~31s | 2.8741/2.3551 | 69.62/46.27 |
| mixed_float16 | ~31s | 2.3916/3.1765 | 61.12/58.58 |

## References

- [Gemma: Google's Open AI Language Models](https://blog.google/technology/developers/gemma-open-models/)
- [Weights & Biases](https://wandb.ai/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [KerasNLP Documentation](https://keras.io/keras_nlp/)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
