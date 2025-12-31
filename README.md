<div align="center">

# DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models

<a href="https://arxiv.org/abs/24XX.XXXXX"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
<a href="https://diffthinker-project.github.io/"><img src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge&logo=github&logoColor=white" alt="Project Page"></a>
<a href="https://huggingface.co/yhx12/DiffThinker"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107?style=for-the-badge&logoColor=white" alt="Models"></a>

<br>

**Code and models will be released gradually today (2025.12.31).**
| Status | Task |
| :---: | :--- |
| ![Released](https://img.shields.io/badge/-%E2%9C%93-brightgreen) | **Training Code** |
| ![Released](https://img.shields.io/badge/-%E2%9C%93-brightgreen) | **FrozenLake** |
| ![Released](https://img.shields.io/badge/-%E2%9C%93-brightgreen) | **Maze** |
| ![Pending](https://img.shields.io/badge/-%20-lightgrey) | **TSP** |
| ![Pending](https://img.shields.io/badge/-%20-lightgrey) | **Sudoku** |
| ![Pending](https://img.shields.io/badge/-%20-lightgrey) | **Jigsaw** |
</div>

### Project Overview



```text
DiffThinker/
├── DiffSynth-Studio/
│   ├── add/
│   │   ├── cmd/
│   │   │   ├── 2509.sh     #train Qwen-Image-Edit-2509
│   │   │   └── 2511.sh     #train Qwen-Image-Edit-2509
│   │   ├── infer/
│   │   │   ├── infer_with_middle
│   │   │   └── infer.py       
│   │   └── merge_ckpt.py
│   ├── diffsynth/
│   └── ...
├── FrozenLake/
├── Maze/                       #example
│   ├── 8_test/                 #test
│   │   ├── 8_1_001.png
│   │   ├── 8_1_001_solution.png
│   │   ├── 8_1_001.txt         #metadata
│   │   ├── ...
│   │   └── path.json           #ground-truth
│   ├── 16_test/                #test
│   ├── 32_test/                #test
│   ├── eval/
│   │   ├── diffthinker.py      #infer
│   │   ├── parse_image.py      #parse
│   │   ├── eval_path.py        #compare with answer
│   │   ├── gen_and_parse.sh
│   │   └── eval_path.sh
│   ├── gen_image.py            #generate dataset
│   ├── gen.txt                 #examples for gen_image.py
│   ├── gen_csv.py              #generate metadata for training
│   ├── ...
├── TSP/
├── Sudoku/
└── Jigsaw/
```

### Quick Start
```
git clone https://github.com/lcqysl/DiffThinker.git
cd DiffThinker/DiffSynth-Studio
pip install -e .
pip install gymnasium

# (Optional) Install vLLM for OCR tasks
# we recommend installing it in a SEPARATE environment to avoid conflicts.
# pip install vllm
```


### Trainging
We use Maze as an example to demonstrate the full pipeline: Data Preparation -> Training -> Inference -> Parsing -> Evaluation.
First, download the **[pre-trained models](https://huggingface.co/yhx12/DiffThinker)**.
```code
cd Maze

# 1. Data Preparation
python gen_image.py --size 8 --num 2000 --min_len 1 --out ./8_train
python gen_csv.py --dir ./8_train

# Note: We recommend following the configurations in Maze/gen.txt 
# to reproduce the difficulty levels used in our paper.

# 2. Training
cd ../DiffSynth-Studio
bash add/cmd/2509.sh
```

#Inference & Evaluation
```code
cd Maze

# 1. Inference and Parsing
bash eval/gen_and_parse.sh

# 2. Evaluation
bash eval/eval_path.sh

# 3. Individual Inference
python ../DiffSynth-Studio/add/infer/infer.py
python ../DiffSynth-Studio/add/infer/infer_with_middle.py
```