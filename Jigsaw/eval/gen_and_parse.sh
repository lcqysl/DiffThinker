MODEL="Qwen-Image-Edit"

python eval/diffthinker.py --level 2x2 --model "${MODEL}"
python eval/diffthinker.py --level 3x3 --model "${MODEL}"
python eval/diffthinker.py --level 4x4 --model "${MODEL}"
python eval/VisPuzzle/diffthinker.py --level VisPuzzle --model "${MODEL}"

python eval/parse_image.py 2x2_test/result --grid 2x2
python eval/parse_image.py 3x3_test/result --grid 3x3
python eval/parse_image.py 4x4_test/result --grid 4x4
python eval/VisPuzzle/parse_image.py VisPuzzle_test/result



