import sys
import pandas as pd
import random
import time
import copy
import os.path as osp

sys.path.append('../')
from base.base_algorithm import BaseAlgorithm
from utils.user_config import save_data, DEFAULT_DATA_DIR


class Mondrian(BaseAlgorithm):

    def __init__(self, algorithm_name, k):
        super().__init__(algorithm_name, k)
        self.df = None
        self.return_list = []
        self.return_data = []
        self.loss_metric = [0, 0]
        self.total_age_num = 0
        self.total_edu_num = 0

    def initial_setting(self):
        """
        Implement initial_setting method that sets reduction pair and pop redundant keys in data set
        depends on different algorithm.
        """
        """pop up redundant key in Mondrian"""
        self.data_set.pop('gender')
        self.data_set.pop('race')
        self.data_set.pop('marital_status')

        """setup dataframe grouped by occupation"""
        occupation_list = []
        index = []
        for j in range(self.data_num):
            self.data_set['education_num'][j] = int(self.data_set['education_num'][j])
            self.data_set['age'][j] = int(self.data_set['age'][j])
            if self.data_set['occupation'][j] not in occupation_list:
                occupation_list.append(self.data_set['occupation'][j])
        self.df = pd.DataFrame(data=self.data_set)

        self.total_age_num = max(self.data_set['age'])+1 - min(self.data_set['age'])
        self.total_edu_num = max(self.data_set['education_num'])+1 - min(self.data_set['education_num'])

    def process(self):
        """
        processing using Mondrian method
        """
        choice_flag = 0
        succe_flag = 1
        temp_list = []
        process_list = []
        new_process_list = []
        print("start processing by using Mondrian")
        time_start = time.time()

        new_process_list.clear()
        new_process_list.append(self.df)
        while True:
            choice_flag = random.randint(0, 1)
            process_list.clear()
            process_list = copy.deepcopy(new_process_list)
            new_process_list.clear()
            count = 0
            print("process list len:{}".format(len(process_list)))
            for data_frame in process_list:
                count += len(data_frame)
            print(count)
            for data_frame in process_list:
                success_flag = 0
                temp_list.clear()
                # print(data_frame.shape[0])
                if choice_flag and data_frame['education_num'].max() != data_frame['education_num'].min():
                    df_1 = data_frame[data_frame['education_num'] > data_frame['education_num'].median()]
                    df_2 = data_frame[data_frame['education_num'] <= data_frame['education_num'].median()]
                    if len(df_1):
                        temp_list.append(df_1)
                    if len(df_2) < len(data_frame):
                        temp_list.append(df_2)
                else:
                    df_1 = data_frame[data_frame['age'] > data_frame['age'].median()]
                    df_2 = data_frame[data_frame['age'] <= data_frame['age'].median()]
                    if len(df_1):
                        temp_list.append(df_1)
                    if len(df_2) < len(data_frame):
                        temp_list.append(df_2)

                if temp_list:
                    success_flag = 1
                    for item in temp_list:
                        if len(item) < self.k:
                            success_flag = 0

                if success_flag == 1:
                    for new_df in temp_list:
                        new_process_list.append(new_df)

                elif success_flag == 0:
                    self.return_list.append(data_frame)

            if not new_process_list:
                break

        end_time = time.time()
        print("processing done, with time {}".format(end_time - time_start))
        print(self.data_num)

    def compute_loss_metric(self):
        """
        compute loss metric and generalize
        :return:
        """
        for df in self.return_list:
            new_age = []
            new_edu_num = []
            max_age = df['age'].max()
            min_age = df['age'].min()
            max_edu_num = df['education_num'].max()
            min_edu_num = df['education_num'].min()
            for j in range(len(df)):
                if min_age == max_age:
                    new_age.append(str(max_age))
                else:
                    new_age.append(str(min_age) + '-' + str(max_age))
                if min_edu_num == max_edu_num:
                    new_edu_num.append(max_edu_num)
                else:
                    new_edu_num.append(str(min_edu_num) + '-' + str(max_edu_num))
            df.iloc[:, 0] = new_age
            df.iloc[:, 1] = new_edu_num
            if max_age != min_age:
                self.loss_metric[0] += len(df) * (max_age - min_age - 1) / (self.total_age_num - 1)
            if min_edu_num != max_edu_num:
                self.loss_metric[1] += len(df) * (max_edu_num - min_edu_num - 1) / (self.total_edu_num - 1)

        self.loss_metric[0] = self.loss_metric[0] / self.data_num
        self.loss_metric[1] = self.loss_metric[1] / self.data_num
        print("loss metric:{}".format(self.loss_metric))

    def save_data(self):
        file_name = 'Mondrian' + '_' + str(self.k) + '.csv'
        file_path = osp.join(DEFAULT_DATA_DIR, file_name)
        ret = self.return_list[0]
        for df in self.return_list[1:]:
            ret = pd.concat([ret, df], axis=0, ignore_index=True)

        save_data(algorithm=self.algorithm_name, k=self.k, data=ret)
