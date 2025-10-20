function [res,resgrad] = ml_errro_radar(x0,y0i,ri,ni,si,di)

res = [(sqrt(sum((y0i-x0).^2))-ri)./si (x0'*ni-sum(ni.*y0i))./(ri.*di)]';
%err = sqrt(res'*res/2);

resgrad = [1./(sqrt(sum((y0i-x0).^2)))'./(si').*(x0-y0i)' ; ni'./ri'./di'];

