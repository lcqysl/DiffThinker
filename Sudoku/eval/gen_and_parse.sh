MODEL="Qwen-Image-Edit"

python eval/diffthinker.py --level 45 --model "${MODEL}"
python eval/diffthinker.py --level 40 --model "${MODEL}"
python eval/diffthinker.py --level 35 --model "${MODEL}"

python eval/parse_image.py 45_test/result
python eval/parse_image.py 40_test/result
python eval/parse_image.py 35_test/result

