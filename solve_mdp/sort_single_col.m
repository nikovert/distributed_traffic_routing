function [B, index] = sort_single_col(A, col, arg)
% sorts array A along dim=2 with respect to col
[r, c] = size(A);

if nargin < 3
    arg = 'ascend';
end
A(:, col) = A(:, col) + rand(r,1)*1e-6; % rand ensures no values are identical
A_sorted = sort(A(:, col), 1, arg);

index = (repmat(A_sorted, [1 r])' ...
            == repmat(A(:, col), [1 r]))' * (1:r)';

B = A(index,:);

end