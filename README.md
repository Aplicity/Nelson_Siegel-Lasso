# Nelson_Siegel-Lasso
基于Lasso回归的Nelson-Siegel扩展模型
## Model
![image](https://github.com/Aplicity/Nelson_Siegel-Lasso/blob/master/model_fig.png)
## DataSet Describe
 自变量为时间𝑡，因变量为利率𝑟。其余字段可以不理。
 * train:训练集，共 66 天，对每天数据进行拟合，得到66×3个回归方程。 
 * test:测试集，共 66 天，用于对上面拟合出的模型进行验证。成绩计算指标：RMSE、MAPE。

## Task
 * 拟合出各个模型的未知参数，并把每天的回归参数结果汇总在一个表中
 * 把所有天数的预测函数图汇总在一起

## Document
 * matlab_sample_script：Lasso回归代码示例
 * Nelson-Siegel：模型一的Matlab代码，{main_model_1.m(主函数)、nelsonsim.m、nelsonfun.m}标准的Nelson-Siegel求解。nonlinear_fit.m利用非线性参数拟合函数求解。
 * Nelson&Lasso：基于Lasso回归的Nelson-Siegel扩展模型。
   - images为拟合效果
   - para_result.csv为拟合参数结果。
 * Reference_paper：参考文献
