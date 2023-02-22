#!/bin/bash
nohup tensorboard --host=0.0.0.0 --port=99  --logdir=/app/logs &

#python /app/main.py --model=train --prename=gpt2-large --epochs=20 --batch_size=2 --input_max_length=512 --output_max_length=128 --lr_max_value=0.00001 --datasets=records_2023-02-17_17-40-47.pkl
#python /app/main.py --model=train --prename=gpt2 --epochs=50 --batch_size=8 --input_max_length=512 --output_max_length=128 --lr_max_value=0.00001 --datasets=records_2023-02-17_17-40-47.pkl

#python /app/main.py --model=train --memory_limit=22 --prename=gpt2-large --epochs=20 --batch_size=2 --input_max_length=512 --output_max_length=128 --lr_max_value=0.000001 --datasets=records_2023-02-17_17-40-47.pkl
#python /app/main.py --model=train --memory_limit=12 --prename=gpt2 --epochs=50 --batch_size=8 --input_max_length=512 --output_max_length=128 --lr_max_value=0.000001 --datasets=records_2023-02-17_17-40-47.pkl

#python /app/main.py --model=train --memory_limit=24 --prename=gpt2-large --epochs=50 --batch_size=2 --input_max_length=512 --output_max_length=128 --lr_max_value=0.000001 --datasets=records_2023-02-17_17-40-47.pkl

python /app/main.py --model=train --memory_limit=24 --prename=gpt2-large --epochs=50 --batch_size=2 --input_max_length=512 --output_max_length=256 --lr_max_value=0.000001 --datasets=records_2023-02-17_17-40-47.pkl
# python /app/model_save.py --memory_limit=12 --tmp_path=tmp --save_epoch_num=4