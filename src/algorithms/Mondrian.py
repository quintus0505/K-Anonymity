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
        self.occupation_list = []
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
        df = pd.DataFrame(data=self.data_set)
        # for i in range(len(occupation_list)):
        #     print(self.data_set['occupation'].count(occupation_list[i]))
        for occupation in occupation_list:
            index.append(occupation)
            occupation_df = df[df['occupation'].isin(index)].reset_index()
            index.clear()
            self.occupation_list.append(occupation_df)

        self.total_age_num = max(self.data_set['age']) - min(self.data_set['age'])
        self.total_edu_num = max(self.data_set['education_num']) - min(self.data_set['education_num'])

    def process(self):
        """
        processing using Mondrian method
        """
        for occupation in self.occupation_list:
            if occupation.shape[0] < self.k:
                print("Cannot find a reduction method, having an occupation only contain {} people".format(
                    occupation.shape[0]))
                return False
        choice_flag = 0
        succe_flag = 1
        temp_list = []
        process_list = []
        new_process_list = []
        print("start processing by using Mondrian")
        time_start = time.time()
        for occupation in self.occupation_list:
            new_process_list.clear()
            new_process_list.append(occupation)
            while True:
                choice_flag = random.randint(0, 1)
                process_list.clear()
                process_list = copy.deepcopy(new_process_list)
                new_process_list.clear()
                for data_frame in process_list:
                    success_flag = 1
                    temp_list.clear()
                    # print(data_frame.shape[0])
                    if choice_flag:
                        df_1 = data_frame[data_frame['education_num'] > data_frame['education_num'].median()]
                        df_2 = data_frame[data_frame['education_num'] <= data_frame['education_num'].median()]
                        temp_list.append(df_1)
                        temp_list.append(df_2)
                    else:
                        df_1 = data_frame[data_frame['age'] > data_frame['age'].median()]
                        df_2 = data_frame[data_frame['age'] <= data_frame['age'].median()]
                        temp_list.append(df_1)
                        temp_list.append(df_2)

                    for item in temp_list:
                        if len(item) < self.k:
                            success_flag = 0
                            break
                    if success_flag == 1:
                        for new_df in temp_list:
                            new_process_list.append(new_df)

                if not new_process_list:
                    for df in process_list:
                        self.return_list.append(df)
                    break

        for item in self.return_list:
            item.drop('index', axis=1, inplace=True)
            item.reset_index()
        end_time = time.time()
        print("processing done, with time {}".format(end_time - time_start))

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

            self.loss_metric[0] += len(df) * (max_age - min_age - 1) / (self.total_age_num - 1)
            self.loss_metric[1] += len(df) * (max_edu_num - min_edu_num - 1) / (self.total_edu_num - 1)

        self.loss_metric[0] = self.loss_metric[0] / self.data_num
        self.loss_metric[1] = self.loss_metric[1] / self.data_num
        print("loss metric:{}".format(self.loss_metric))

    def save_data(self):
        file_name = 'Mondrian' + '_' + str(self.k) + '.csv'
        file_path = osp.join(DEFAULT_DATA_DIR, file_name)
        ret = self.return_list[0]
        for df in self.return_list:
            ret = pd.concat([ret, df], axis=0, ignore_index=True)

        save_data(algorithm=self.algorithm_name, k=self.k, data=ret)
