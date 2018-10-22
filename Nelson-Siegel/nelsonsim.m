function[par]=nelsonsim(x,y)
par.tau=fminbnd(@(tau)nelson(tau),0,10);
par.beta=betas(par.tau);
function[f]=nelson(tau)
[b,f]=betas(tau);
end
    function[b,varargout]=betas(tau)
        i=x(:)/tau;
        j=1-exp(-i);
        n=length(x);
        z=[ones(n,1),j./i,(j./i)+j-1];
        b=(z'*z)\(z'*y(:));
        e=y(:)-z*b;
        varargout(1)={e'*e};
    end
end


