clc;clear
[Numeric,Txt]=xlsread('./train/2014331.xlsx');
a=Numeric(:,1);
c=x2mdate(a);
Settle=datenum('3/31/2014');
ZeroRates=Numeric(:,11);
y=ZeroRates';
d=datestr(c);
x=date2time(Settle,c,1);
y=y';
par=nelsonsim(x,y); % ģ�����ֵ
p=nelsonfun(x,par); % ��Ͻ����tauΪlamda��betaΪ�ع�ϵ��
