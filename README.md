<div align="center">

# DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models

<a href="https://arxiv.org/abs/24XX.XXXXX"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
<a href="https://lcqysl.github.io/DiffThinker"><img src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge&logo=github&logoColor=white" alt="Project Page"></a>
<a href="https://huggingface.co/yhx12/DiffThinker"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107?style=for-the-badge&logoColor=white" alt="Models"></a>

<br>

**Code and models will be released gradually today (2025.12.31).**

</div>
<br>

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
│
├── FrozenLake/
├── Maze/                       #example
│   ├── 8_test/                 #test
│   ├── 16_test/                #test
│   ├── 32_test/                #test
│   ├── eval/
│   │   └── diffthinker.py      #infer
│   │   └── parse_image.py      #parse
│   │   └── eval_path.py        #compare with answer
│   │   └── gen_and_parse.sh
│   │   └── eval_path.sh
│   ├── gen_image.py            #generate dataset
│   ├── gen_csv.py              #generate metadata for training
│   ├── ...
├── TSP/
├── Sudoku/
└── Jigsaw/
```



