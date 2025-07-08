import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

tips = sns.load_dataset("tips")
# print(tips.head())
print(tips.describe())

print("Mean:",tips['total_bill'].mean())
print("Median:",tips['total_bill'].median())
print("Mode:",tips['total_bill'].mode()[0])

sns.histplot(tips['total_bill'],kde=True,color='skyblue',bins=30)
plt.title("Total bill distribution")
plt.show()

stats.probplot(tips['total_bill'],dist='norm',plot=plt)
plt.title("QQ plot")
plt.show()

print("Skewness:",stats.skew(tips['total_bill']))
print("kurtosis:",stats.kurtosis(tips['total_bill']))

weekend = tips[tips['day'].isin(['Sat','Sun'])]['tip']
weekday = tips[tips['day'].isin(['Thur','Fri'])]['tip']

t_stat , p_val = stats.ttest_ind(weekend,weekday)
print(f"T statistics:{t_stat:.4f},P-value:{p_val:.4f}")

from scipy.stats import f_oneway

sun = tips[tips['day'] == 'Sun']['tip']
sat = tips[tips['day'] == 'Sat']['tip']
thur = tips[tips['day'] == 'Thur']['tip']
fri = tips[tips['day'] == 'Fri']['tip']

f_stat, p_value = f_oneway(sun, sat, thur, fri)
print(f"F-statistic: {f_stat:.4f}, P-value: {p_value:.4f}")


mean = tips['total_bill'].mean()
std = tips['total_bill'].std()
n = len(tips['total_bill'])

# 95% CI
conf_int = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(n))
print("95% Confidence Interval for Total Bill:", conf_int)

correlation = tips['total_bill'].corr(tips['tip'])
print("Correlation between total bill and tip:", correlation)
