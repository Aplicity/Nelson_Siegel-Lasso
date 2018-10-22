clc;clear
train_fileName = 'train/2013131.xlsx'
train_data = xlsread(train_fileName);
train_x = train_data(:,10);
train_y = train_data(:,end);

test_fileName = 'test/12013131.xlsx'
test_data = xlsread(test_fileName);
test_x = test_data(:,10);
test_y = test_data(:,end);

syms t 
f=fittype('b0+b1*(1-exp(-t/a))./(t/a)+b2*((1-exp(-t/a))./(t/a)-exp(-t/a))','independent','t','coefficients',{'b0','b1','b2','a'});


cfun=fit(train_x,train_y,f) 

y_pred=cfun(test_x);
train_y_pred = cfun(train_x);

error_square = (test_y - y_pred).^2;
mse = mean(error_square);
rmse = sqrt(mse)
MAPE = mean(abs(test_y-y_pred)./test_y)

figure
scatter(train_x,train_y)
hold on 
plot(train_x,train_y_pred)
