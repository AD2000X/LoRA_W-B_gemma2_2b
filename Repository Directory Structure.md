```bash
gemma2-lora-finetune/
│
├── LICENSE                       # Apache 2.0 license
├── README.md                     # Project description
├── requirements.txt              # Dependencies
├── setup.py                      # Installation script
├── main.py                       # Main program entry point
│
├── src/                          # Source code directory
│   ├── __init__.py               # Makes src an importable module
│   ├── data.py                   # Data loading and processing
│   ├── model.py                  # Model definition and training functions
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation script
│
├── notebooks/                    # Jupyter notebooks directory
│   ├── lora_finetune_fp32.ipynb         # FP32 precision training
│   ├── lora_finetune_bfloat16.ipynb     # BFloat16 precision training
│   └── lora_finetune_float16.ipynb      # Float16 precision training
│
├── models/                       # Model storage directory
│   └── .gitkeep                  # Ensures empty directory is included in Git
│
└── data/                         # Data directory
    └── .gitkeep                  # Ensures empty directory is included in Git
    
