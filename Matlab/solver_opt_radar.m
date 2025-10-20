function [R,sols,res] = solver_opt_radar(y0i,ri,ni,si,di,use_eigenvector)
% Finds all stationary points to
% min_x sum_j w(j) * (|x-s(:,k)|^2 - d(k)^2)^2

if nargin < 6
    use_eigenvector = 1;
end

n = 3;
%n = size(yi,1);

w = 1./(8*ri.^2.*si.^2);
g = 1./(2*ri.^2.*di.^2);

w = w(:);
g = g(:);

ri = ri(:);
% Normalize weights
%w = w / sum(w);
nw = sum(w);
w = w/nw;
g = g/nw;


% shift center
t = sum(y0i .* w(:,ones(1,n))',2);
yi = y0i - t * ones(1,size(y0i,2));




% Construct A,b such that (x'*x)*x + A*x + b = 0
% ws2md2 = w.*(sum(yi.^2,1).'-ri.^2);
% A = 2*(yi.* w(:,ones(n,1)).') * yi.' + sum(ws2md2) * eye(n);
% b = -yi*ws2md2;

wy2mr2 = w.*(sum(yi.^2,1).'-ri.^2);
A = 2*(yi.* w(:,ones(n,1)).') * yi.' + sum(wy2mr2) * eye(n) + 0.5*(ni.* g(:,ones(n,1)).') * ni.';
b = -yi*wy2mr2 -0.5*ni*(g'.*sum(ni.*yi,1)).';


[V,D] = eig(A);
bb = V'*b;

% basis = [x^2,y^2,z^2,x,y,z,1]
AM = [-D diag(-bb) zeros(n,1);
     zeros(n,n) -D -bb;
     ones(1,n) zeros(1,n+1)];

if use_eigenvector
    [VV,DD] = eig(AM);
    VV = VV ./ (ones(2*n+1,1)*VV(end,:));
    ro = V*VV(n+1:2*n,:);
else
    DD = eig(AM);
    % eigenvector-less solution extraction
    ro = zeros(n,2*n+1);
    for k = 1:2*n+1
        z = [zeros(n,n);-eye(n)];
        T = AM - DD(k)*eye(2*n+1);
        ro(:,k) = (T(:,1:end-1).' \ z).'*T(:,end);
    end
    ro = V*ro;
end


% perform some refinement on the roots
for i = 1:2*n+1    
    roi = ro(:,i);
    if max(abs(imag(roi))) > 1e-6
        continue
    end
    for k = 1:3
        res = (roi'*roi)*roi + A*roi + b;
        if norm(res) < 1e-8
            break;
        end
        J = (roi'*roi)*eye(n) + 2 *(roi*roi') + A;
        roi = roi - J\res;
    end
    ro(:,i) = roi;
end

% Revert translation of coordinate system
sols = ro + t*ones(1,size(ro,2));

% find best stationary point
cost = inf;
R = zeros(n,1);
for k = 1:size(sols,2)
    if sum(abs(imag(sols(:,k)))) > 1e-6
        continue;
    end
    rok = real(sols(:,k));
    % cost_k = sum(w'.*(sum((rk(:,ones(1,size(s0,2)))-s0).^2)- d'.^2).^2);
    cost_k = sum(w'.*(sum((rok(:,ones(1,size(y0i,2)))-y0i).^2)- ri'.^2).^2) + ...
        sum(g'.*sum(ni.*rok-ni.*yi,1));
    
    if cost_k < cost
        cost = cost_k;
        R = rok;
    end
end

if nargout == 3
    lambda = diag(DD);
    res = [];
    for k = 1:2*n+1
        rok = ro(:,k);
        lambdak = lambda(k);
        res(:,k) = [(rok.'*rok)*rok + A*rok + b; lambdak - rok.'*rok];
    end
end


