import os
import re
import datetime
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from keras.engine import data_adapter
# from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

from utils.save_data_tool import save_pkl

PWD = os.getcwd()


def train_val_split(samples, train_val_split_rate=0.1, seed=5):
    """
    json对象多个字段分割训练集和验证集
    :param samples: json对象{'input_word_ids':[...], 'input_mask':[...], 'input_type_ids':[...], 'label':[...]}
    :type samples: object
    :param train_val_split_rate: 测试集占总样本的比率
    :type train_val_split_rate: float
    :param seed: 随机数生成器的种子
    :type seed: int
    :return: 返回训练集和验证集
    :rtype: (
                {'input_word_ids':[...], 'input_mask':[...], 'input_type_ids':[...], 'label':[...]},
                {'input_word_ids':[...], 'input_mask':[...], 'input_type_ids':[...], 'label':[...]}
            )
    """
    np.random.seed(seed)
    num_samples = len(samples[list(samples.keys())[0]])
    train_samples_len = int(np.round(num_samples * (1 - train_val_split_rate)))

    samples_idx = np.random.permutation(np.arange(num_samples))
    train_samples_idx = samples_idx[:train_samples_len]
    val_samples_idx = samples_idx[train_samples_len:]

    train_samples = {}
    val_samples = {}

    for key in samples.keys():
        train_samples[key] = np.array([samples[key][int(idx)] for idx in train_samples_idx])
        val_samples[key] = np.array([samples[key][int(idx)] for idx in val_samples_idx])

    return train_samples, val_samples


# class GPT2Customize(TFGPT2LMHeadModel):
class GPT2Customize(TFT5ForConditionalGeneration):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Run forward pass.
        with tf.GradientTape() as tape:
            output = self(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                decoder_input_ids=x['decoder_input_ids'],
                labels=y,
                training=True)
            loss = output.loss
            logits = output.logits
            self.compute_loss(x['input_ids'], y, logits, sample_weight)
        # self._validate_target_and_loss(y, loss)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return self.compute_metrics(x['input_ids'], y, logits, sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        output = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=x['decoder_input_ids'],
            labels=y,
            training=False)
        logits = output.logits

        # Updates stateful loss metrics.
        self.compute_loss(x['input_ids'], y, logits, sample_weight)

        return self.compute_metrics(x['input_ids'], y, logits, sample_weight)


class GPT2ConditionalGeneration:
    def __init__(self,
                 memory_limit=22,
                 load_weight_latest=False,
                 # pre_m_name_or_path="gpt2-large",
                 # pre_m_name_or_path="gpt2",
                 pre_m_name_or_path="t5-small",
                 epochs=10,
                 batch_size=32,
                 input_max_length=512,
                 output_max_length=256,
                 lr_max_value=0.005,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 datasets_name='records_2023-02-17_17-40-47.pkl'):

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

        self.load_weight_latest = load_weight_latest
        self.pre_m_name_or_path = pre_m_name_or_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.lr_max_value = lr_max_value
        self.datasets_name = datasets_name

        pre_w_path = os.path.join(PWD, "pretrained_w", pre_m_name_or_path)

        with tf.device("/CPU:0"):
            # self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
            self.gpt2_tokenizer = T5Tokenizer.from_pretrained(
                pre_m_name_or_path,
                model_max_length=input_max_length,
                cache_dir=os.path.join(pre_w_path, "tokenizer"))

        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

        self.gpt2_model = GPT2Customize.from_pretrained(
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

    def dataset_add_letter_body(self, datasetes):
        for index in range(len(datasetes)):
            datasetes[index]['letter_body'] = list(
                filter(lambda x: len(x.strip().split()) > 20, datasetes[index]['recommendation_letter']))

    def input_label_generate(self, introduction_start, reasons_end, reasons_1, detailed_2):
        inputs = []
        labels = []

        for item in introduction_start:
            input_sent = f"My name is {item['teacher_name']}. I am {item['job_title']} of {item['office']}. {item['student_name']} and I have a {item['relationship']} relationship. Below are the details of our relationship: {item['introduction']}"
            label_sent = item['letter_body'][0]

            inputs.append(input_sent)
            labels.append(label_sent)

        for item in reasons_end:
            input_sent = f"Applicant's name is {item['student_name']}. The reason for the recommendation is {item['reasons']}"
            label_sent = item['letter_body'][-1]

            inputs.append(input_sent)
            labels.append(label_sent)

        for item in reasons_1:
            input_sent = f"Applicant's name is {item['student_name']}. {item['student_name']}`s degree is {item['student_statuses'][0]['degree']}. {item['student_name']}`s major is {item['student_statuses'][0]['major']}. The reason for the recommendation is {item['reasons']}"
            label_sent = item['letter_body'][1]

            inputs.append(input_sent)
            labels.append(label_sent)

        for item in detailed_2:
            input_sent = f"Applicant's name is {item['student_name']}. {item['student_name']}`s project skills is {item['detailed']}"
            label_sent = item['letter_body'][2]

            inputs.append(input_sent)
            labels.append(label_sent)

        return inputs, labels

    def to_token(self, inputs, labels):
        with tf.device("/CPU:0"):
            input_mask_ids = self.gpt2_tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=self.input_max_length,
                return_tensors="tf")

            label_ids = self.gpt2_tokenizer(
                labels,
                padding=True,
                truncation=True,
                return_attention_mask=False,
                max_length=self.output_max_length,
                return_tensors="tf").input_ids

        input_numpy_ids = input_mask_ids.input_ids.numpy()
        label_numpy_ids = label_ids.numpy()

        pad_token_id = self.gpt2_tokenizer.pad_token_id

        input_ids_analyze = [list(filter(lambda col: col != pad_token_id, row)) for row in input_numpy_ids]
        label_ids_analyze = [list(filter(lambda col: col != pad_token_id, row)) for row in label_numpy_ids]

        def generator_decoder_input_id(item):
            return [self.gpt2_model.config.decoder_start_token_id] + item[: len(item) - 1]

        decoder_input_ids = list(map(generator_decoder_input_id, label_ids_analyze))
        decoder_input_ids_tensor = tf.ragged.constant(decoder_input_ids, dtype=label_ids.dtype).to_tensor()

        # sample_size = label_ids.shape[0]
        # input_token_size = input_mask_ids.input_ids.shape[1]
        # label_token_size = label_ids.shape[1]
        # eos_token_id = self.gpt2_tokenizer.eos_token_id
        #
        # padding_matrix = tf.fill((sample_size, input_token_size - label_token_size), eos_token_id)
        # label_ids = tf.concat([label_ids, padding_matrix], axis=-1)

        if not os.path.exists("analyze"):
            os.makedirs("analyze")

        input_ids_analyze_len = list(map(lambda x: len(x), input_ids_analyze))
        label_ids_analyze_len = list(map(lambda x: len(x), label_ids_analyze))
        input_ids_analyze_len.sort(reverse=True)
        label_ids_analyze_len.sort(reverse=True)

        plt.figure(figsize=(24, 10), dpi=200)
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

        ax.plot(list(range(len(input_ids_analyze_len))), input_ids_analyze_len, label="input_ids_len")
        ax.plot(list(range(len(label_ids_analyze_len))), label_ids_analyze_len, label="label_ids_len")

        plt.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
        plt.savefig(os.path.join("analyze", "token_len.png"))

        return input_mask_ids, label_ids, decoder_input_ids_tensor

    def dataset_processing(self, datasetes):
        self.dataset_add_letter_body(datasetes)

        introduction = list(
            filter(lambda x: re.search(r"When do students enter the University", x['introduction'], re.I) is None,
                   datasetes))
        introduction = list(filter(lambda x: len(x['introduction'].strip().split()) >= 10, introduction))
        introduction_start = list(filter(lambda x: len(x['letter_body']) >= 4, introduction))
        introduction_start = list(filter(lambda x: len(x['letter_body'][0].strip().split()) >= 30, introduction_start))

        reasons = list(filter(lambda x: len(x['reasons'].strip().split()) >= 10, datasetes))
        reasons_end = list(filter(lambda x: len(x['letter_body']) >= 4, reasons))
        reasons_end = list(filter(lambda x: len(x['letter_body'][-1].strip().split()) >= 30, reasons_end))

        reasons_1 = list(filter(lambda x: len(x['letter_body'][1].strip().split()) >= 30, reasons_end))

        detailed = list(filter(lambda x: len(x['detailed'].strip().split()) >= 10, datasetes))
        detailed_2 = list(filter(lambda x: len(x['letter_body']) >= 4, detailed))
        detailed_2 = list(filter(lambda x: len(x['letter_body'][2].strip().split()) >= 30, detailed_2))

        inputs, labels = self.input_label_generate(introduction_start, reasons_end, reasons_1, detailed_2)

        input_mask_ids, label_ids, decoder_input_ids = self.to_token(inputs, labels)

        return input_mask_ids.input_ids, input_mask_ids.attention_mask, label_ids, decoder_input_ids

    def __call__(self):
        datasetes = np.load(os.path.join(PWD, 'dataset', self.datasets_name), allow_pickle=True)
        input_ids, attention_mask, label_ids, decoder_input_ids = self.dataset_processing(datasetes)

        train_sample, val_sample = train_val_split({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "decoder_input_ids": decoder_input_ids
        })

        tmp_path = './tmp'
        checkpoint_filepath = tmp_path + '/cp-{epoch:04d}.ckpt'
        if self.load_weight_latest:
            latest = tf.train.latest_checkpoint(tmp_path)
            # latest = "./tmp/cp-0028.ckpt"
            print(f"\033[0;34mlatest:\033[0;35m{latest}\033[0m")
            self.gpt2_model.load_weights(latest)
        else:
            os.system(f'rm -rfv {tmp_path}')

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        save_pkl(
            os.path.join(tmp_path, 'config.pkl'),
            {
                'pre_m_name_or_path': self.pre_m_name_or_path,
                'batch_size': self.batch_size,
                'input_max_length': self.input_max_length,
                'output_max_length': self.output_max_length,
                'lr_max_value': self.lr_max_value,
                'datasets_name': self.datasets_name
            })

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_best_only=False,
            verbose=1,
            save_freq='epoch')

        self.gpt2_model.compile(
            optimizer=self.optimizer,
            loss=self.loss_object,
            metrics=[self.sparse_categorical_accuracy])

        self.gpt2_model.fit(
            x={
                "input_ids": train_sample['input_ids'],
                "attention_mask": train_sample['attention_mask'],
                "decoder_input_ids": train_sample['decoder_input_ids']
            },
            y=train_sample['label_ids'],
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(
                {
                    "input_ids": val_sample['input_ids'],
                    "attention_mask": val_sample['attention_mask'],
                    "decoder_input_ids": val_sample['decoder_input_ids']
                },
                val_sample['label_ids']
            ),
            callbacks=[self.tensorboard_callback, model_checkpoint_callback]
        )


if __name__ == "__main__":
    gpt2_conditional_generation = GPT2ConditionalGeneration()
    gpt2_conditional_generation()
