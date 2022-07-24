function [costJ, policy, res_J] = valueIteration(g, P, alpha, NUM_ITER, convergence_tol, costJ)
%% Value Iteration
% Initialize variables that we update during value iteration.
% Cost (here it really is the reward):

nx = size(g,1);

% Policy
policy = zeros(1,nx);    

if nargin < 4
    NUM_ITER = 1000;
end
if nargin < 5
    convergence_tol = 1e-3;
end
if nargin < 6
    costJ = zeros(1,nx);
end

for k=1:NUM_ITER
    costJ_old = costJ;
    for i=1:nx   % loop over two states
        % One value iteration step for each state.
        [costJ(i),policy(i)] = min( squeeze(sum(double(g(i,:,:)).*P(i,:,:), 2))' + alpha*costJ*squeeze(P(i,:,:)) );
    end

    % Save results for plotting later.
    res_J(k,:) = costJ;

    % Construct string to be displayed later:
    dispstr = ['k=',num2str(k,'%5d')];
    for i=1:nx
        dispstr = [dispstr, '   J(',num2str(i),')=',num2str(costJ(i),'%6.4f')];
    end
    
    if abs(costJ_old - costJ) < convergence_tol
        disp(['converged after ',num2str(k), ' iterations']);
        break
    end
    
    % Display:
    disp(dispstr);
end
end
