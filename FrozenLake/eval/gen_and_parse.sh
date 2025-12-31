MODEL="Qwen-Image-Edit"

python eval/vsp/diffthinker.py --level 3 --model "${MODEL}"
python eval/vsp/diffthinker.py --level 4 --model "${MODEL}"
python eval/vsp/diffthinker.py --level 5 --model "${MODEL}"
python eval/vsp/diffthinker.py --level 6 --model "${MODEL}"
python eval/vsp/diffthinker.py --level 7 --model "${MODEL}"
python eval/vsp/diffthinker.py --level 8 --model "${MODEL}"
python eval/vsp_super/diffthinker.py --level 16 --model "${MODEL}"
python eval/vsp_super/diffthinker.py --level 32 --model "${MODEL}"
python eval/parse_image.py