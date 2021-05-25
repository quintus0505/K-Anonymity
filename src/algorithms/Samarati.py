import sys
import time
import copy
from functools import cmp_to_key
import pandas as pd
import os.path as osp
from utils.user_config import save_data, DEFAULT_DATA_DIR

sys.path.append('../')
from base.base_algorithm import BaseAlgorithm
from utils.user_config import save_data, mycmp1


class Samarati(BaseAlgorithm):

    def __init__(self, algorithm_name, k, maxsup):
        super().__init__(algorithm_name, k)
        self.maxsup = maxsup
        # reduction pair
        self.gender_reduction_pair = dict()
        self.race_reduction_pair = dict()
        self.age_reduction_pair = dict()
        self.marital_status_reduction_pair = dict()
        # hierarchy tree
        self.hierarchy_tree = dict()
        self.h = 0
        self.return_dataset = dict()
        self.return_loss_metric = [1, 1, 1, 1]
        # flag for finding solution
        self.global_flag = 0

    def initial_setting(self):
        """
        Implement initial_setting method that sets reduction pair and pop redundant keys in data set
        depends on different algorithm.
        """
        """pop up redundant key in Samarati"""
        self.data_set.pop('education_num')

        """initial reduction_pair"""
        self.gender_reduction_pair = {'Male': '*', 'Female': '*'}
        self.race_reduction_pair = {'White': '*', 'Black': '*', 'Asian-Pac-Islander': '*', 'Amer-Indian-Eskimo': '*',
                                    'Other': '*'}
        self.marital_status_reduction_pair = {'Never-married': 'NM', 'Married-civ-spouse': 'Married',
                                              'Married-AF-spouse': 'Married', 'Divorced': 'leave',
                                              'Separated': 'leave', 'Widowed': 'alone',
                                              'Married-spouse-absent': 'alone',
                                              'NM': '*', 'Married': '*', 'leave': '*', 'alone': '*'}

        for j in range(15, 91, 5):
            for i in range(j, j + 5):
                self.age_reduction_pair[str(i)] = str(j) + '-' + str(j + 4)

        for j in range(10, 91, 10):
            for i in range(j, j + 10, 5):
                self.age_reduction_pair[str(i) + '-' + str(i + 4)] = str(j) + '-' + str(j + 9)

        for j in range(10, 101, 20):
            for i in range(j, j + 20, 10):
                self.age_reduction_pair[str(i) + '-' + str(i + 9)] = str(j) + '-' + str(j + 19)

            self.age_reduction_pair[str(j) + '-' + str(j + 19)] = '*'
        """
        initial hierarchy_tree
        Demonstrating reduction tree in hierarchy, the original dataset is represented as [0,0,0,0], 
        the all masked dataset is represented as [1,1,2,4], which means that the keys representing 
        each elements in the list are 'gender','race','marital-status' and 'age' respectively.
        """
        self.hierarchy_tree['0'] = [[0, 0, 0, 0]]
        self.hierarchy_tree['1'] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        self.hierarchy_tree['2'] = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],
                                    [0, 0, 1, 1], [0, 0, 2, 0], [0, 0, 0, 2]]

        self.hierarchy_tree['3'] = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 2, 0], [1, 0, 0, 2],
                                    [0, 1, 2, 0], [0, 1, 0, 2], [0, 0, 1, 2], [0, 0, 2, 1], [0, 0, 0, 3]]

        self.hierarchy_tree['4'] = [[1, 1, 1, 1], [1, 1, 2, 0], [1, 1, 0, 2], [1, 0, 2, 1], [1, 0, 1, 2], [1, 0, 0, 3],
                                    [0, 1, 2, 1], [0, 1, 1, 2], [0, 1, 0, 3], [0, 0, 1, 3], [0, 0, 2, 2], [0, 0, 0, 4]]

        self.hierarchy_tree['5'] = [[1, 1, 1, 2], [1, 1, 2, 1], [1, 1, 0, 3], [1, 0, 2, 2], [1, 0, 1, 3], [1, 0, 0, 4],
                                    [0, 1, 2, 2], [0, 1, 1, 3], [0, 1, 0, 4], [0, 0, 2, 3], [0, 0, 1, 4]]

        self.hierarchy_tree['6'] = [[1, 1, 2, 2], [1, 1, 1, 3], [1, 1, 0, 4], [1, 0, 2, 3], [1, 0, 1, 4],
                                    [0, 1, 2, 3], [0, 1, 1, 4], [0, 0, 2, 4]]

        self.hierarchy_tree['7'] = [[1, 1, 2, 3], [1, 1, 1, 4], [1, 0, 2, 4], [0, 1, 2, 4]]
        self.hierarchy_tree['8'] = [[1, 1, 2, 4]]

        for i in range(9):  # set hierarchy_tree element in order which has increase loss_metric
            self.hierarchy_tree[str(i)].sort(key=cmp_to_key(mycmp1), reverse=True)

        '''set height of the tree'''
        self.h = len(self.hierarchy_tree) - 1

    def process(self):
        """
        main function using Samarati algorithm
        """
        count = 0  # number of data needs to be suppressed
        current_layer = int(self.h / 2)  # current height in reduction tree

        switch_height = current_layer
        current_reduction_vector = None
        flag = False  # find a ideal data set or not
        print("start processing by using Samarati")
        time_start = time.time()
        while True:
            flag, self.return_loss_metric, current_reduction_vector, current_layer, switch_height, suppression_num = self.function(
                current_layer, switch_height)

            if switch_height == 0:
                end_time = time.time()
                break
            if flag:
                self.global_flag = 1
                print("loss_metric:{}".format(self.return_loss_metric))
                print("reduction_vector:{}".format(current_reduction_vector))
                print("suppression_num:{}".format(suppression_num))
        if self.global_flag:
            print("find solution")
        else:
            print("cannot find a reduction method")

        total_time = end_time - time_start
        print("total_time:{}".format(total_time))

    def function(self, current_layer, switch_height):
        """
        reduce data set
        :param current_layer:
        :param switch_height:
        :return: whether the original dataset can achieve K-Anonymity under given k and maxsup
                 and processed data_set
        """

        for distance_vector in self.hierarchy_tree[str(current_layer)]:
            # copy temp data set
            # print("current_layer:{} distant_vector:{}".format(current_layer, distance_vector))
            temp_data_set = copy.deepcopy(self.data_set)
            loss_metric = [1, 1, 1, 1]
            # reduce data set
            for i in range(distance_vector[0]):
                for j in range(self.data_num):
                    gender = temp_data_set['gender'][j]
                    temp_data_set['gender'][j] = self.gender_reduction_pair[gender]

            for i in range(distance_vector[1]):
                for j in range(self.data_num):
                    race = temp_data_set['race'][j]
                    temp_data_set['race'][j] = self.race_reduction_pair[race]

            for i in range(distance_vector[2]):
                for j in range(self.data_num):
                    marital_status = temp_data_set['marital_status'][j]
                    temp_data_set['marital_status'][j] = self.marital_status_reduction_pair[marital_status]

            for i in range(distance_vector[3]):
                for j in range(self.data_num):
                    age = temp_data_set['age'][j]
                    temp_data_set['age'][j] = self.age_reduction_pair[age]

            flag, suppression_key, suppression_num = self.judge(data_set=temp_data_set)
            """if find ideal reduction, return reduced dataset and loss metric"""

            if flag:
                for j in range(self.data_num):
                    temp_list = [temp_data_set['gender'][j], temp_data_set['race'][j],
                                 temp_data_set['marital_status'][j], temp_data_set['age'][j]]
                    if temp_list in suppression_key:
                        temp_data_set['gender'][j] = '*'
                        temp_data_set['race'][j] = '*'
                        temp_data_set['marital_status'][j] = '*'
                        temp_data_set['age'][j] = '*'
                        temp_data_set['occupation'][j] = '*'

                loss_metric = self.compute_loss_metric(distance_vector, temp_data_set)
                switch_height = switch_height = int(switch_height / 2)
                current_layer = current_layer - switch_height
                self.return_dataset = copy.deepcopy(temp_data_set)
                return True, loss_metric, distance_vector, current_layer, switch_height, suppression_num

            else:
                continue
        switch_height = switch_height = int(switch_height / 2)
        current_layer = current_layer + switch_height
        return False, None, None, current_layer, switch_height, None

    def judge(self, data_set):
        """
        judge whether data_set achieve K-Anonymity
        :param data_set: processed data set
        :return: whether dataset achieve K-Anonymity, suppression_key and suppress_num for suppression
        """
        judging_set = []  # transform dict to list
        judging_key = []  # key of all the data
        suppression_key = []
        suppress_num = 0  # data needs to be suppressed
        for j in range(self.data_num):
            judging_set.append(
                [data_set['gender'][j], data_set['race'][j], data_set['marital_status'][j], data_set['age'][j]
                 ])

        for j in range(self.data_num):
            if judging_set[j] not in judging_key:
                judging_key.append(judging_set[j])

        for key in judging_key:
            if judging_set.count(key) < self.k:
                suppress_num += judging_set.count(key)
                suppression_key.append(key)
            if suppress_num > self.maxsup:
                return False, None, None

        return True, suppression_key, suppress_num

    def compute_loss_metric(self, distance_vector, temp_data_set):
        """
        :param distance_vector: element in self.hierarchy_tree, represent reduction status
        :param temp_data_set: temp_data_set
        :return: loss_metric, figures in this list represent the loss metric of 'gender','race',
                'marital-status' and 'age' respectively.
        """
        loss_metric = [0.0, 0.0, 0.0, 0.0]
        for j in range(self.data_num):
            if (distance_vector[0] == 0 and temp_data_set['gender'][j] == '*') or distance_vector[0] == 1:
                loss_metric[0] += 1.0
            if (distance_vector[1] == 0 and temp_data_set['race'][j] == '*') or distance_vector[1] == 1:
                loss_metric[1] += 1.0
            if (distance_vector[2] == 0 and temp_data_set['marital_status'][j] == '*') or distance_vector[2] == 2:
                loss_metric[2] += 1.0
            elif distance_vector[2] == 1:
                if temp_data_set['marital_status'][j] != 'NM':
                    loss_metric[2] += float(1 / 6)

            if (distance_vector[3] == 0 and temp_data_set['age'][j] == '*') or distance_vector[3] == 4:
                loss_metric[3] += 1
            elif distance_vector[3] == 1:
                loss_metric[3] += float(4 / 79)
            elif distance_vector[3] == 2:
                loss_metric[3] += float(9 / 79)
            elif distance_vector[3] == 3:
                loss_metric[3] += float(19 / 79)

        for i in range(len(loss_metric)):
            loss_metric[i] = loss_metric[i] / self.data_num

        return loss_metric

    def save_data(self):
        """
        save data, change dict() to str(list)  (redundant?)
        """

        df = pd.DataFrame(self.return_dataset)
        save_data(algorithm=self.algorithm_name, k=self.k, maxsup=self.maxsup, data=df)
