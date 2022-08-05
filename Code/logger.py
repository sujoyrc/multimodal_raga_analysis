from time import localtime, strftime
from read_yaml import read_config
import os
import numpy as np
import csv
import pandas as pd


def flatten_dict(dd, separator='-', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}


class Logger:

    def __init__(self, config, prefix=''):
        self.log_path = os.path.join(config['dir']['log_results'], prefix + '_log.txt')
        self.log_file = open(self.log_path, 'a')

        self.res_path = os.path.join(config['dir']['log_results'], prefix + '_results.csv')
        # columns in CSV
        self.column_keys = list(flatten_dict(config).keys())
        self.eval = ['val_test', 'pearson', 'f_score_2', 'f_score_3', 'time']
        self.column_keys += ['eval:' + x for x in self.eval]

        if not os.path.exists(self.res_path):
            with open(self.res_path, 'w') as f:
                f.write(','.join(self.column_keys) + '\n')

    def print(self, *args):
        """
        Print to terminal and also write to log file
        :param args: what to print
        """
        s = '[' + strftime("%Y-%m-%d %H:%M:%S", localtime()) + '] '
        s += ' '.join([str(x) for x in args])
        print(s)
        self.log_file.write(s + '\n')

    def make_section(self, heading):
        to_append = str(heading) + ',' + ',' * (len(self.column_keys) - 2) + '\n'
        # print(to_append)
        with open(self.res_path, 'a') as f:
            f.write(to_append)

    def append_row(self, config, val_or_test, p_f2_f3):
        """
        Add a new row to CSV file
        :param config: config file
        :param val_or_test: whether 4-fold CV or 3 outer-fold testing
        :param p_f2_f3: list of pearson, F-score(>=2), F-score(>=3) values
        :return:
        """
        if not isinstance(p_f2_f3[0], list):
            p_f2_f3 = [p_f2_f3]
        params = list(flatten_dict(config).values())
        for i in range(len(params)):
            if isinstance(params[i], list):
                params[i] = '_'.join([str(x) for x in params[i]])
            else:
                params[i] = str(params[i])
        params.append(val_or_test)

        for res_list in p_f2_f3:
            mean, std = np.mean(res_list), np.std(res_list)
            params.append('{} +- {}'.format(np.round(mean, 3), np.round(std, 4)))

        # track time
        params.append(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        # add row
        to_append = ','.join(params) + '\n'
        # print(to_append)
        with open(self.res_path, 'a') as f:
            f.write(to_append)
        # pd.read_csv(self.res_path, header=None).T.to_csv(self.res_path[:-4]+'_T.csv', header=False, index=False)


if __name__ == '__main__':
    config = read_config()
    a = Logger(config)
    a.append_row(config, 'val', [[0.879, 0.7, 0.8345], [0.879, 0.7, 0.2], [0.9, 0.3, 0.8345]])
    config['num_layers'] = 3
    a.make_section('hello')
    # a.print('hello')
    # a.print(6)
