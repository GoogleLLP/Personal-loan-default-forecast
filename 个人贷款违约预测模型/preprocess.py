# %%
from read_file import read_file
from fix_money import fix_money
import datetime
import pandas as pd


# %% 读取文件
creat_var = locals()
files = read_file()
for var_name, var_value in files.items():
    creat_var[var_name] = var_value

# %% 对loans进行预处理
loans["date"] = pd.to_datetime(loans["date"])
loans["status"] = loans["status"].map({"B": 1, "D": 1, "C": 2, "A": 0})

# %% 对trans进行预处理
trans["date"] = pd.to_datetime(trans["date"])
trans[["amount", "balance"]] = trans[["amount", "balance"]].applymap(fix_money)
temp_trans = trans.groupby(["account_id", "type"])[["amount"]].sum()
trans_amount = pd.pivot_table(temp_trans, values="amount", index="account_id", columns="type")
trans_amount["r_out_in"] = trans_amount["借"] / trans_amount["贷"]
trans_amount.fillna(0, inplace=True)

trans_balance = trans.groupby("account_id")["balance"].agg([("mean", "mean"), ("std", "std")])
trans_balance["cv"] = trans_balance["std"] / trans_balance["mean"]

trans = pd.merge(trans, trans_balance, left_on="account_id", right_index=True, how="left")
trans = pd.merge(trans, trans_amount, left_on="account_id", right_index=True, how="left")

trans["r_lb"] = trans["amount"] / trans["mean"]
trans["r_lincome"] = trans["amount"] / trans["贷"]

# %% 合并loans和trans并处理
data = pd.merge(loans.drop("amount", axis=1), trans, on="account_id", how="left")
data = data[(data["date_x"] > data["date_y"]) & (data["date_x"] < data["date_y"]+datetime.timedelta(days=365))]
data.drop(["date_x", "date_y", "trans_id"], inplace=True, axis=1)

# %% 对accounts进行预处理并合并
accounts["date"] = pd.to_datetime(accounts["date"])
accounts.drop("date", inplace=True, axis=1)
accounts = pd.get_dummies(accounts, columns=["frequency"])
data = pd.merge(data, accounts, on="account_id", how="left")

# %% 合并district并处理
data = pd.merge(data, district, left_on="district_id", right_on="A1", how="left")
data.drop("district_id", inplace=True, axis=1)

# %% 对disp进行预处理并合并
disp["owner"] = disp["type"].map({"所有者": 1, "用户": 0})
disp.drop("type", inplace=True, axis=1)
data = pd.merge(data, disp, on="account_id", how="left")
data.drop(["account_id", "disp_id"], inplace=True, axis=1)

# %% 对clients进行预处理并合并
clients["sex"] = clients["sex"].map({"女": 0, "男": 1})
clients["birth_date"] = pd.to_datetime(clients["birth_date"])
data = pd.merge(data, clients, on="client_id", how="left")
data.drop("client_id", inplace=True, axis=1)
data = pd.get_dummies(data, columns=["operation"])
data = data[(data["owner"] == 1) & (data["status"] != 2)]







# %% 特征选择
# loans: status, duration
# district: 全部
# trans: mean, std, cv, 贷, 借, r_out_in, r_lb, r_lincome


