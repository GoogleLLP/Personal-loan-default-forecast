
# coding: utf-8

# In[ ]:

import pandas as pd
import os
##########################################################################################################################################################################################
# # 1 数据整理

# ## 1.1导入数据

# In[2]:
os.chdir("data/")
loanfile = os.listdir()
createVar = locals()
for i in loanfile:
    if i.endswith("csv"):        
        createVar['t_'+ i.split('.')[0]] = pd.read_csv(i, encoding = 'gbk')
        print(i.split('.')[0])


# ## 1.2、生成被解释变量bad_good
        
# In[3]:

bad_good = {'B':1, 'D':1, 'A':0, 'C': 2}
t_loans['bad_good'] = t_loans.status.map(bad_good) 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html

t_loans.head()

# ## 1.3、借款人的年龄、性别

# In[4]:

data2 = pd.merge(t_loans, t_disp, on = 'account_id', how = 'left') # ‘left’, ‘right’, ‘outer’, ‘inner’
data2 = pd.merge(data2, t_clients, on = 'client_id', how = 'left')
data2=data2[data2.type=='所有者']

data2.head()
# ## 1.4、借款人居住地的经济状况

# In[5]:

data3 = pd.merge(data2, t_district, left_on = 'district_id', right_on = 'A1', how = 'left')
data3.head()

# ## 1.5、贷款前一年内的账户平均余额、余额的标准差、变异系数、平均收入和平均支出的比例

# In[6]:

data_4temp1 = pd.merge(t_loans[['account_id', 'date']],
                       t_trans[['account_id','type','amount','balance','date']],
                       on = 'account_id')
data_4temp1.columns = ['account_id', 'date', 'type', 'amount', 'balance', 't_date']
data_4temp1 = data_4temp1.sort_values(by = ['account_id','t_date'])

data_4temp1['date']=pd.to_datetime(data_4temp1['date'])
data_4temp1['t_date']=pd.to_datetime(data_4temp1['t_date'])
data_4temp1.tail()

# ## 将对账户余额进行清洗
# In[9]:

data_4temp1['balance2'] = data_4temp1['balance'].map(
    lambda x: int(''.join(x[1:].split(','))))
data_4temp1['amount2'] = data_4temp1['amount'].map(
    lambda x: int(''.join(x[1:].split(','))))

data_4temp1.head()

# ## 根据取数窗口提取交易数据--选取贷款放款日之前一年的交易信息！
# In[10]:

import datetime
data_4temp2 = data_4temp1[data_4temp1.date>data_4temp1.t_date][
    data_4temp1.date<data_4temp1.t_date+datetime.timedelta(days=365)]
data_4temp2.tail()

# ### 1.5.1账户平均余额、余额的标准差、变异系数

# In[11]:

data_4temp3 = data_4temp2.groupby('account_id')['balance2'].agg([('avg_balance','mean'), ('stdev_balance','std')])
data_4temp3['cv_balance'] = data_4temp3[['avg_balance','stdev_balance']].apply(lambda x: x[1]/x[0],axis = 1)
#https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.Series.apply.html

#data_4temp3['cv_balance2'] = data_4temp3['stdev_balance'] / data_4temp3['avg_balance']

data_4temp3.head()

# ### 1.5.2 平均支出和平均收入的比例
# In[12]:

type_dict = {'借':'out','贷':'income'}
data_4temp2['type1'] = data_4temp2.type.map(type_dict)
data_4temp4 = data_4temp2.groupby(['account_id','type1'])[['amount2']].sum()
data_4temp4.head()


# In[13]:

data_4temp5 = pd.pivot_table(
    data_4temp4, values = 'amount2', 
    index = 'account_id', columns = 'type1')
# https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.pivot_table.html

data_4temp5.fillna(0, inplace = True)
data_4temp5['r_out_in'] = data_4temp5[
    ['out','income']].apply(lambda x: x[0]/x[1], axis = 1)
data_4temp5.head()


# In[14]:

data4 = pd.merge(data3, data_4temp3, left_on='account_id', right_index= True, how = 'left')
data4 = pd.merge(data4, data_4temp5, left_on='account_id', right_index= True, how = 'left')

data4.head()
# ## 1.6、计算贷存比，贷收比

# In[15]:

data4['r_lb'] = data4[['amount','avg_balance']].apply(lambda x: x[0]/x[1],axis = 1)
data4['r_lincome'] = data4[['amount','income']].apply(lambda x: x[0]/x[1],axis = 1)

data4.head()
# In[17]:
##########################################################################################################################################################################################
# # 2 构建Logistic模型
data4.columns
# •提取状态为C的用于预测。其它样本随机抽样，建立训练集与测试集

# In[17]:

data_model=data4[data4.status!='C']
for_predict=data4[data4.status=='C']

# 利用sample划分
train = data_model.sample(frac=0.7, random_state=1235).copy()
test = data_model[~ data_model.index.isin(train.index)].copy()
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))

candidates = ['bad_good', 'A1', 'GDP', 'A4', 'A10', 'A11', 'A12','amount', 'duration',
       'A13', 'A14', 'A15', 'a16', 'avg_balance', 'stdev_balance',
       'cv_balance', 'income', 'out', 'r_out_in', 'r_lb', 'r_lincome']
data_for_select = train[candidates]

#%% ---使用train_test_split--生成train和test

# In[22]:

import statsmodels.api as sm
import statsmodels.formula.api as smf

# 过滤式特征选择---前向选择(forward selection)和后向淘汰(backward elimination)
def forward_select(data, response):

    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data, 
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data, 
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)

#%% 使用sklearn中方法实现特征选择 并测试结果
#from sklearn.linear_model import LogisticRegression   
#from sklearn.feature_selection import RFE


# In[23]:

lg_m1 = forward_select(data=data_for_select, response='bad_good')
lg_m1.summary().tables[1]


# In[24]:

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
fpr, tpr, th = metrics.roc_curve(test.bad_good, lg_m1.predict(test))
plt.figure(figsize=[6, 6])
plt.plot(fpr, tpr, 'b--')
plt.title('ROC curve')
plt.show()


# In[25]:

print('AUC = %.4f' %metrics.auc(fpr, tpr))


# In[28]:

for_predict['prob']=lg_m1.predict(for_predict)
for_predict[['account_id','prob']].head()


# In[ ]:




