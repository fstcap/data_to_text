#!/bin/bash
nohup tensorboard --host=0.0.0.0 --port=99  --logdir=/app/logs &
python /app/main.py --model=train --prename=gpt2-large --epochs=20 --batch_size=2 --input_max_length=512 --output_max_length=128 --lr_max_value=0.00001 --datasets=records_2023-02-17_17-40-47.pkl