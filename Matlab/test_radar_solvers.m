%% generate some random data

sc = 100;
x0 = randn(3,1)*sc; % 3D target point

s0 = 0.1; % standard deviation of distance measurements
d0 = 0.5/360*2*pi; % angle error

N = 15; % number of radar positions 

% random positions of radars and their measurements to the unknown x0
yi = zeros(3,N); 
ni0 = zeros(3,N);

for iii = 1:N
    kul = true;
    while kul
        [R,~] = qr(randn(3,1));
        t = randn(3,1)*sc;
        P = [R -R*t];
        u = P*[x0;1];
        kul = u(3)<0;
    end
    yi(:,iii) = t; % radar position
    v = cross(R(2,:)',x0-t); 
    ni0(:,iii) = v/norm(v); % radar heading

end

r0i = sqrt(sum((x0-yi).^2)); % radar distance to x0
ri = r0i+s0*randn(1,N); % radar distance measurement 

ni = ni0+randn(size(ni0))*d0; % radar bearing measurement
ni = ni./sqrt(sum(ni.^2));

si = s0*ones(1,N); % stds of distances 
di = d0*ones(1,N); % stds of bearings

[R0,sols0,res0] = solver_opt_radar(yi,r0i,ni0,si,di);  % ML solution without noise
[R,sols,res] = solver_opt_radar(yi,ri,ni,si,di); % ML solution with noise
[R0lin,res0lin] = solver_radar_linear(yi,r0i,ni0);  % Linear solution without noise
[Rlin,reslin] = solver_radar_linear(yi,ri,ni); % Linear solution with noise

disp('    GT    ML(no noise)    ML   Lin(no noise)  Lin :')
disp([x0 R0 R R0lin Rlin])
disp('Distance to ground truth for ')
disp('     ML       Linear')
disp([norm(x0-R) norm(x0-Rlin)])

