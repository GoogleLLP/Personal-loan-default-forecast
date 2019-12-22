# %% 导入包
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime


# %% 定义一个绘图函数
def draw(pic, i=1):
    plt.figure(i)
    pic
    plt.show()


# %% 定义一个函数
def fix_money(money):
    result = money.replace("$", '')
    result = result.replace(",", '')
    return result


# %% 读取数据
createVar = locals()
files = os.listdir("data/")
for file in files:
    if file.endswith(".csv"):
        createVar[file.split(".")[0]] = pd.read_csv("data/"+file, encoding="gbk", low_memory=False)

# %% 数据预处理
data = card.merge(disp, how="left", on="disp_id")
data = data.merge(clients, how="left", on="client_id")
data = data.merge(trans, how="left", on="account_id")
data[["issued", "birth_date", "date"]] = data[["issued", "birth_date", "date"]].apply(pd.to_datetime)
data["age"] = (data["issued"] - data["birth_date"]).apply(lambda x: x.days//365)
data[["amount", "balance"]] = data[["amount", "balance"]].applymap(fix_money)
data[["amount", "balance"]] = data[["amount", "balance"]].apply(pd.to_numeric, errors="ignore")


# %% 不同类型卡的持卡人的性别对比
card_sex_count = pd.crosstab(data["type_x"], data["sex"])
card_sex_count_plot = card_sex_count.plot(kind="bar", stacked=True)
draw(card_sex_count_plot, 1)

card_sex_ratio = card_sex_count.apply(lambda x: x/x.sum(), axis=1)
card_sex_ratio_plot = card_sex_ratio.plot(kind="bar", stacked=True)
draw(card_sex_ratio_plot, 2)

# %% 不同类型卡的持卡人在办卡时的平均年龄对比
card_age_plot = data.groupby("type_x")[["age"]].agg("mean").plot(kind="bar")
draw(card_age_plot, 3)

card_age_box = data.boxplot(by="type_x", column="age")
draw(card_age_box, 4)

# %% 不同类型卡的持卡人在办卡前一年内的平均帐户余额对比
data_last_year = data[
    (data["issued"] > data["date"]) &
    (data["issued"] < data["date"] + datetime.timedelta(days=365))
]
card_amount_plot = data_last_year.boxplot(by="type_x", column="amount")
draw(card_amount_plot, 5)

# %% 不同类型卡的持卡人在办卡前一年内的平均收入对比
temp_balance = data_last_year.groupby(["type_x", "client_id"])[["balance"]].mean()
balance_last_pivot = pd.pivot_table(data=temp_balance, index="type_x", columns="client_id", values="balance")
balance_last_pivot.fillna(0, inplace=True)
plt.figure(6)
# 如果传入plt.boxplot是一个ndarray，则按ndarray每一列画
plt.boxplot(balance_last_pivot.T.values)
plt.show()
plt.figure(7)
# 如果传入plt.boxplot是一个DataFrame，则按DataFrame.index画
plt.boxplot(balance_last_pivot)
plt.show()


