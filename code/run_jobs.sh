#!/bin/bash

python3 training.py BBBC022
python3 predict_generator.py BBBC022
bash cp_res.sh BBBC022/0324_sample_size_500/

python3 training.py BBBC022_10
python3 predict_generator.py BBBC022_10
bash cp_res.sh BBBC022_10/0324_sample_size_500/

python3 training.py BBBC022_100
python3 predict_generator.py BBBC022_100
bash cp_res.sh BBBC022_100/0324_sample_size_500/

python3 training.py BBBC022_1000
python3 predict_generator.py BBBC022_1000
bash cp_res.sh BBBC022_1000/0324_sample_size_500/



