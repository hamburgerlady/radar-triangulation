function x0 = modbundle_sparse(x0,y0i,ri,ni,si,di,iter,lambda)
% optimering av \sum_j ||f_ij(P_i,U_j)-u_j||_2^2 
% genom linearisering av f_ij(P_i,U_j)
% iter - antal iterationer
% lambda - dämpfaktor


%res = compute_res(P,U,u);    
[res,resgrad] = ml_errro_radar(x0,y0i,ri,ni,si,di);

%fprintf('%d\t%f\t%f',0,res'*res,lambda);
i = 0;
while i <= iter
    A = resgrad;
    B = res;
    %[A,B] = setup_lin_system(P,U,u);  
    %res = compute_res(P,U,u);
    res = ml_errro_radar(x0,y0i,ri,ni,si,di);
    
        C = (A'*A+lambda*speye(size(A,2),size(A,2)));
        c = A'*B;
        %fprintf('\t\t\t\t\t\t\tSolving mod-Newton system.');
        d = -C\c;
    %fprintf('\tDone.\n');

    xnew = x0+d;
    %[Pnew,Unew] = update_var(d,P,U);
    %resnew = compute_res(Pnew,Unew,u);
    resnew = ml_errro_radar(xnew,y0i,ri,ni,si,di);
    
    i = i+1;
    while (resnew'*resnew) > (res'*res)
        lambda = lambda*2;
        C = (A'*A+lambda*speye(size(A,2),size(A,2)));
        c = A'*B;
        %fprintf('\t\t\t\t\t\t\tSolving mod-Newton system.');
        d = -C\c;
        %fprintf('\tDone.\n');

        %[Pnew,Unew] = update_var(d,P,U);
        
        %resnew = compute_res(Pnew,Unew,u);

        xnew = x0+d;
        resnew = ml_errro_radar(xnew,y0i,ri,ni,si,di);
        i = i + 1;
    end
    %if lambda > 0.001
        lambda = lambda/1.25;
    %end
    x0 = xnew;
    %U = Unew;
    %P = Pnew;
    %fprintf('%d\t%f\t%f',i,resnew'*resnew,lambda);
end
%fprintf('\n');
