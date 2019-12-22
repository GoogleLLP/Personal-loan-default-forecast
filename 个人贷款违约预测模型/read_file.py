import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# 读取指定目录下同类型文件
def read_file(path="data/", type="csv"):
    files = os.listdir(path)
    result = {}
    for file in files:
        filename = file.split(".")[0]
        filetype = file.split(".")[-1]
        if filetype == type:
            # print(path+file)
            result[filename] = pd.read_csv(path + file, encoding="gbk")
            # file_var[filename] = pd.read_csv(path + file, encoding="gbk")
    return result


# 第一个返回编码后的数值，第二个返回对应的列名
def one_hot(series):
    enc = OneHotEncoder()
    return enc.fit_transform(series.values.reshape(-1, 1)).toarray(), enc.get_feature_names()


if __name__ == "__main__":
    creat_var = locals()
    files = read_file()
    for var_name, var_value in files.items():
        creat_var[var_name] = var_value
    print(accounts)
