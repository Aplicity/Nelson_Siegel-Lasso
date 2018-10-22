function[y]=nelsonfun(x,par)
i=x(:)/par.tau;
j=1-exp(-i);
y=par.beta(1)+par.beta(2)*j./i+par.beta(3)*((j./i)+j-1);
end