import os
import re
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.save_data_tool import save_pkl


def check_array(arr, index1, index2, index3, null_count):
    try:
        return arr[index1][index2][index3].strip()
    except IndexError:
        null_count[0] += 1
        arr[index1].insert(index2, [''])
        return ''


class DataAnalyze(object):
    def __init__(self, path="./original_source/rl_tables_2023-02-15_14-23-16.pkl"):
        self.datasets = np.load(path, allow_pickle=True)
        print(f"\033[0;32m{len(self.datasets)}\033[0m")

    def array_to_records(self):
        records = []
        continue_count = 0
        null_count = [0]
        error_introduction_count = 0

        count_start = 0
        count_end = 0
        for index in range(0, len(self.datasets)):
            dataset = self.datasets[index]
            count_start += 1

            info = dict()
            info['order_id'] = dataset['order_id']
            info['student_name'] = dataset['table'][1][1][0]
            info['student_gender'] = dataset['table'][1][2][0]

            student_statuses = []
            student_status = dict()
            for i in range(2, len(dataset['table'])):

                key_name = dataset['table'][i][0][0].strip()
                if key_name == 'degree':
                    student_status = dict()
                    student_status['degree'] = dataset['table'][i][1][0]
                    student_status['major'] = check_array(dataset['table'], i, 3, 0, null_count)
                elif key_name == 'university' or key_name == 'School':
                    student_status['university'] = dataset['table'][i][1][0].strip()
                    student_status['Attendance time'] = dataset['table'][i][3][0].strip()
                    student_statuses.append(student_status)
                else:
                    break

            if len(student_statuses) == 0:
                continue_count += 1
                continue

            info['student_statuses'] = student_statuses

            info['teacher_name'] = dataset['table'][5 + (len(student_statuses) - 1) * 2][1][0].strip()
            info['teacher_gender'] = dataset['table'][5 + (len(student_statuses) - 1) * 2][3][0].strip()
            info['job_title'] = check_array(dataset['table'], 6 + (len(student_statuses) - 1) * 2, 1, 0, null_count)
            info['office'] = dataset['table'][6 + (len(student_statuses) - 1) * 2][3][0].strip()
            info['unit'] = dataset['table'][7 + (len(student_statuses) - 1) * 2][1][0].strip()
            info['datetime'] = dataset['table'][7 + (len(student_statuses) - 1) * 2][3][0].strip()
            info['relationship'] = dataset['table'][8 + (len(student_statuses) - 1) * 2][1][0].strip()

            info['relationship'] = dataset['table'][8 + (len(student_statuses) - 1) * 2][1][0].strip()

            info['introduction'] = ('/n'.join(dataset['table'][10 + (len(student_statuses) - 1) * 2][0])).strip()
            info['reasons'] = ('/n'.join(dataset['table'][12 + (len(student_statuses) - 1) * 2][0])).strip()
            info['detailed'] = ('/n'.join(dataset['table'][14 + (len(student_statuses) - 1) * 2][0])).strip()
            info['conclusion'] = ('/n'.join(dataset['table'][16 + (len(student_statuses) - 1) * 2][0])).strip()

            rl = list(filter(lambda x: x.strip() != '' and x is not None, dataset['rl']))
            info['recommendation_letter'] = rl

            if re.search(r"When do students enter the University", info['introduction'], re.I) is not None:
                error_introduction_count += 1

            records.append(info)
            count_end += 1

        dataset_path = "./dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        save_pkl(os.path.join(dataset_path, f"records_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"), records)
        print()
        return records

    def count_sent_len(self, records):
        record_countes = []
        for record in records:
            count_dict = dict()
            for key in record.keys():
                if key == 'student_statuses':
                    count_dict[key] = len(record[key])
                elif key == 'recommendation_letter':
                    letter_textes = []
                    for sent in record[key]:
                        if len(sent.strip().split()) > 20:
                            letter_textes.append(sent)
                    count_dict['letter_paragraph_count'] = len(letter_textes)
                    if len(letter_textes) > 1:
                        count_dict['letter_start_sent_len'] = len(letter_textes[0].strip().split())
                        count_dict['letter_end_sent_len'] = len(letter_textes[-1].strip().split())
                    else:
                        count_dict['letter_start_sent_len'] = 0
                        count_dict['letter_end_sent_len'] = 0
                else:
                    count_dict[key] = len(record[key].split())
            record_countes.append(count_dict)
        return record_countes

    def record_count_to_list(self, record_countes):
        data_frame = pd.DataFrame(record_countes)
        count_list = data_frame.to_dict(orient='list')
        return count_list

    def plot_count(self, data, limit_len=100):
        root_path = "./plot_count"
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        plt.figure(figsize=(24, 10), dpi=200)
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

        for key, value in data.items():
            value = list(value)
            value.sort(reverse=True)
            for index in range(len(value)):
                if value[index] >= limit_len:
                    value[index] = limit_len
                else:
                    break
            ax.plot(list(range(len(value))), value, label=key)
        plt.legend(loc=2, bbox_to_anchor=(1.0, 1.162))
        plt.savefig(os.path.join(root_path, "plot_sent_len.png"))

        for key, value in data.items():
            plt.figure(figsize=(24, 10), dpi=200)
            ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

            value = list(value)
            value.sort(reverse=True)

            for index in range(len(value)):
                if value[index] >= limit_len:
                    value[index] = limit_len
                else:
                    break

            ax.plot(list(range(len(value))), value, label=key)
            plt.legend(loc=2, bbox_to_anchor=(1.0, 1.162))
            plt.savefig(os.path.join(root_path, f"{key}.png"))

    def __call__(self, *args, **kwargs):
        records = self.array_to_records()
        record_countes = self.count_sent_len(records)
        count_list = self.record_count_to_list(record_countes)
        self.plot_count(count_list)


if __name__ == "__main__":
    data_analyze = DataAnalyze()
    data_analyze()
