import os
import datetime

import numpy as np
import tensorflow as tf

from keras.engine import data_adapter
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

PWD = os.getcwd()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_soft_device_placement(True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=22528)])

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class GPT2Customize(TFGPT2LMHeadModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Run forward pass.
        with tf.GradientTape as tape:
            output = self(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                label=y,
                training=True)
            loss = output.loss
            logits = output.logits
            self.compute_loss(x['input_ids'], y, logits, sample_weight)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x['input_ids'], y, logits, sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        output = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            label=y,
            training=False)
        logits = output.logits

        # Updates stateful loss metrics.
        self.compute_loss(x['input_ids'], y, logits, sample_weight)
        return self.compute_metrics(x['input_ids'], y, logits, sample_weight)


class GPT2ConditionalGeneration:
    def __init__(self,
                 pre_m_name_or_path="gpt2-large",
                 epochs=10,
                 batch_size=32,
                 input_max_length=512,
                 output_max_length=512,
                 lr_max_value=0.005,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 datasets_name='rl_tables_2023-02-15_14-23-16.pkl'):
        pre_w_path = os.path.join(PWD, "pretrained_w", pre_m_name_or_path)
        with tf.device("/CPU:0"):
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                pre_m_name_or_path,
                model_max_length=input_max_length,
                cache_dir=os.path.join(pre_w_path, "tokenizer"))

        self.GPT2Model = GPT2Customize.from_pretrained(
            pre_m_name_or_path,
            cache_dir=os.path.join(pre_w_path, "model"))

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(
            lr_max_value,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon)

        self.sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='SparseCategoricalAccuracy')

        log_dir = f"logs/{pre_m_name_or_path}/epochs{epochs}_batch_size{batch_size}_input_max_length{input_max_length}_output_max_length{output_max_length}_lr_max_value{lr_max_value}/{datasets_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log_absolute_dir = os.path.join(PWD, log_dir)

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_absolute_dir, histogram_freq=1)

    def dataset_processing(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    pass
