#!/bin/bash

# if you only have a single GPU, you can run the following command
# python3 train.py --model transformer --accelerator gpu --gpu 0 --folds "0-15"

python3 train.py --model transformer --accelerator gpu --gpu 0 --folds "0-7"   &
python3 train.py --model transformer --accelerator gpu --gpu 1 --folds "7-15"  &
wait
