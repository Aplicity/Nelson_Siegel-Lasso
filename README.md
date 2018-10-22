# Nelson_Siegel-Lasso
基于Lasso回归的Nelson-Siegel扩展模型

## DataSet Describe
 自变量为时间𝑡，因变量为利率𝑟。其余字段可以不理。
 * train:训练集，共 66 天，对每天数据进行拟合，得到66×3个回归方程。 
 * test:测试集，共 66 天，用于对上面拟合出的模型进行验证。成绩计算指标：RMSE、MAPE。

## Task
 * 拟合出各个模型的未知参数，并把每天的回归参数结果汇总在一个表中
 * 把所有天数的预测函数图汇总在一起
