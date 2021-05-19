import os.path as osp
import pickle
from functools import cmp_to_key

DEFAULT_DATASET_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(osp.dirname(__file__)))), 'dataset')
DEFAULT_DATA_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(osp.dirname(__file__)))), 'data')
DEFAULT_ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(osp.dirname(__file__))))


def load_data(file_name='adult.data', dataset_dir=DEFAULT_DATASET_DIR):
    """
    load_data():load data from file

    :arg file_name: Name of the dataset to load.
    :arg dataset_dir: Directory of data
    :return: original dataset

    there are 15 labels in adult.data:
    'age', 'work_class', 'final_weight', 'education','education_num',
    'marital_status', 'occupation', 'relationship','race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'class'.
    In this lab work, only 'age', 'gender', 'race','marital_status',
    'education_num' and 'occupation' are used in this toy example
    """

    data_path = osp.join(dataset_dir, file_name)
    dataset = dict()
    dataset['age'] = []
    dataset['gender'] = []
    dataset['race'] = []
    dataset['education_num'] = []
    dataset['marital_status'] = []
    dataset['occupation'] = []

    with open(data_path, 'rb') as f:
        file_data = f.readlines()

    for i in range(len(file_data)):
        input_str = str(file_data[i]).strip('b\'\\r\\n')
        if "?" not in input_str:
            """strip off info with '?' """
            age, _, _, _, education_num, marital_status, occupation, _, race, gender, _, _, _, _, _ \
                = input_str.split(',')
            dataset['age'].append(age.strip(' '))
            dataset['gender'].append(gender.strip(' '))
            dataset['race'].append(race.strip(' '))
            dataset['education_num'].append(education_num.strip(' '))
            dataset['marital_status'].append(marital_status.strip(' '))
            dataset['occupation'].append(occupation.strip(' '))
    f.close()
    return dataset


def save_data(algorithm, k=0, maxsup=0, data=None, data_dir=DEFAULT_DATA_DIR):
    """
    save_data():save processed dataset

    :arg algorithm: Name of the algorithm.
    :arg k: K-Anonymity parameter k
    :arg maxsup: K-Anonymity algorithm Samarati parameter maxSup
    :arg data: Saved data.
    :arg data_dir: Directory of data

    target dataset containing 'age', 'race', 'gender', 'marital_status', 'occupation' when choosing Samarati
    target dataset containing 'age', 'education_num', 'occupation' when choosing Mondrian
    """
    assert (algorithm in {'Samarati', 'Mondrian'}), "algorithm should be 'Samarati' or 'Mondrian'"

    file_name = None
    if algorithm == 'Samarati':
        file_name = algorithm + '_' + str(k) + '_' + str(maxsup) + '.txt'

    elif algorithm == 'Mondrian':
        file_name = algorithm + '_' + str(k) + '.txt'

    with open(osp.join(data_dir, file_name), 'w') as f:
        f.write(str(data))

    f.close()


def mycmp1(i, j):
    if i[3] != j[3]:
        return i[3] - j[3]
    elif i[2] != j[2]:
        return i[2] - j[2]
    elif i[1] != j[1]:
        return i[1] - j[1]
    else:
        return i[0] - j[0]


if __name__ == '__main__':
    # data_set = load_data()
    #
    # list = []
    # test = dict()
    #
    # # for i in data_test['age']:
    # #     if i not in list:
    # #         list.append(i)
    # for j in range(100):
    #     list.append(
    #         [data_set['gender'][j], data_set['race'][j], data_set['marital_status'][j], data_set['age'][j]])
    # print(list)
    #
    # for i in range(100):
    #     print(list.count(list[i]))
    # # print(data_test['age'].count('90'))
    # # list.sort()
    # # print(list)

    hierarchy_tree = [[1, 1, 1, 1], [1, 1, 2, 0], [1, 1, 0, 2], [1, 0, 2, 1], [1, 0, 1, 2], [1, 0, 0, 3],
                      [0, 1, 2, 1], [0, 1, 1, 2], [0, 1, 0, 3], [0, 0, 1, 3], [0, 0, 2, 2], [0, 0, 0, 4]]
    hierarchy_tree.sort(key=cmp_to_key(mycmp1), reverse=True)
    print(hierarchy_tree)
