function [r,res] = solver_radar_linear(yi,ri,ni)

% Linear approximate solver for
   
    % Subtract mean sender positions.
    yim = mean(yi,2);
    yi2m = mean(dot(yi,yi));
    rim = mean(ri,2);
    A1 = 2*(yim-yi)';
    b1 = -dot(yi,yi)'+yi2m+ri'.^2-rim'.^2;

    A2 = ni';
    b2 = sum(ni.*yi)';

    A = [A1;A2];
    b = [b1;b2];

    r = A\b;
    
    if nargin > 1
        res = A*r-b;
    end
end

