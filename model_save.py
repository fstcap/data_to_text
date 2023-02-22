import os
import datetime
import argparse

import numpy as np
import tensorflow as tf
from transformers import TFGPT2LMHeadModel


class ModelSaveToPretraind:
    def __init__(self, memory_limit=22, tmp_path='tmp', save_epoch_num='1,2,3,4'):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.set_soft_device_placement(True)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit * 1024)])

                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        self.checkpoint_filepath = tmp_path + '/cp-{epoch:04d}.ckpt'
        self.save_epoch_num = [int(num.strip()) for num in save_epoch_num.split(',')]

        config = np.load(os.path.join(tmp_path, 'config.pkl'), allow_pickle=True)

        self.pre_m_name_or_path = config['pre_m_name_or_path']
        self.batch_size = config['batch_size']
        self.lr_max_value = config['lr_max_value']
        self.input_max_length = config['input_max_length']
        self.output_max_length = config['output_max_length']
        self.datasets_name = config['datasets_name']

        self.optimizer = tf.keras.optimizers.Adam(
            self.lr_max_value,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8)

        self.PWD = os.getcwd()
        pretrained_w_path = os.path.join(self.PWD, "pretrained_w")

        self.gpt2_model = TFGPT2LMHeadModel.from_pretrained(
            self.pre_m_name_or_path,
            cache_dir=os.path.join(pretrained_w_path, "model"))

    def __call__(self, *args, **kwargs):
        print("\033[0;31msave start\033[0m")
        for num in self.save_epoch_num:
            self.gpt2_model.load_weights(self.checkpoint_filepath.format(epoch=num))

            checkpoint_dir = f"checkpoints/{self.pre_m_name_or_path}/epochs_{num}_batch_size_{self.batch_size}_lr_max_value_{self.lr_max_value}_input_max_length{self.input_max_length}_output_max_length{self.output_max_length}/{self.datasets_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            print(f"\033[0;34mcheckpoint_dir\033[0;35m{checkpoint_dir}\033[0m")
            self.gpt2_model.save_pretrained(os.path.join(os.path.join(self.PWD, checkpoint_dir)))
        print("\033[0;32msave end\033[0m")


parser = argparse.ArgumentParser(description='select run way')
parser.add_argument("--memory_limit", type=int, default=22)  # 正整数.
parser.add_argument("--tmp_path", type=str, default='tmp')  # 字符串.
parser.add_argument("--save_epoch_num", type=str, default='4,5,6')  #

args = parser.parse_args()

if __name__ == "__main__":
    model_save_to_pretraind = ModelSaveToPretraind(
        memory_limit=args.memory_limit,
        tmp_path=args.tmp_path,
        save_epoch_num=args.save_epoch_num)
    model_save_to_pretraind()
