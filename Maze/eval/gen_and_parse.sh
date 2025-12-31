MODEL="Qwen-Image-Edit"

python eval/diffthinker.py --level 8 --model "${MODEL}"
python eval/diffthinker.py --level 16 --model "${MODEL}"
python eval/diffthinker.py --level 32 --model "${MODEL}"

python eval/parse_image.py --level 8
python eval/parse_image.py --level 16
python eval/parse_image.py --level 32