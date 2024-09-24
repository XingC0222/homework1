import matplotlib.pyplot as plt

# 设置字体,以支持中文显示
plt.rcParams['font.family'] = 'SimHei'

# 年份数据
years = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
         2022, 2023]
#房价数据
prices = [6252.5, 7149.8, 8387, 10425.5, 11813.1, 12900.9, 14964, 17188.8, 19024.7, 21134.6, 22926, 24779.1, 27041.2,
          29883, 33106, 35445.13, 35943.25, 41045.6, 41540.9, 43760.7]

# 定义线性回归函数，通过最小二乘法计算线性回归的斜率和截距
def linear_regression(years, prices):
    n = len(years)
    sum_x = sum(years)
    sum_y = sum(prices)
    sum_xy = sum(x * y for x, y in zip(years, prices))
    sum_xx = sum(x ** 2 for x in years)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept


# 调用线性回归函数，获取斜率和截距，形成预测模型。
slope, intercept = linear_regression(years, prices)

# 生成预测值
predicted_prices = [slope * year + intercept for year in years]

# 绘制图形
plt.figure(figsize=(10, 5))
plt.scatter(years, prices, label='真实房价', color='blue',s=20)  # 使用散点图显示真实值
plt.plot(years, predicted_prices, label='预测房价', linestyle='-', color='red')
plt.title('房价预测模型')
plt.xlabel('年份')
plt.ylabel('价格')
plt.legend()
plt.grid()
plt.show()
