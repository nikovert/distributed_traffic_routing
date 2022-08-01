%% Discount factor.
convergence_tol = 1e-6;
compute_true = true;
use_congestion = true;
alpha = 1;
%% Value Iteration
% Initialize variables that we update during value iteration.
% Cost (here it really is the reward):
if compute_true
    tic;
    [costJ_base, policy_base] = valueIteration(g, P, alpha, 10000, convergence_tol);
    graphPlot = plot_optimalEdge(G, policy_base);
    toc
    
    if use_congestion
        alpha = 0.5;
        congestion = min(1, 0.25 + rand(size(g)));
        g_congested = g .* congestion;
    
        tic;
        [costJ, policy] = valueIteration(g_congested, P, alpha, 10000, convergence_tol);
        toc
    else
        costJ = costJ_base;
    end

    % display the optained costs
    disp('Result:');
    
    % Construct string to be displayed later:
    dispstr = '';
    for i=1:nx
        dispstr = [dispstr, '   J(',num2str(i),')=',num2str(costJ(i),'%6.4f')];
    end
    
    % Display:
    disp(dispstr);
    
    % Plot Value function
    figure;
    stem(1:nx, costJ);
    hold off;
    grid;
    
    axis = gca;
    axis.YLim = [min(costJ)-10, max(costJ)+10];
end

% labelnode(graphPlot,1:length(costJ),costJ)
%%
I_init = I;
%% Aggregate Problem
clear agent in_extra_args
I = I_init;

p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    p = parpool('threads');
end

message_pool = parallel.pool.DataQueue;
afterEach(message_pool, @relay);

tic
agent = Composite([2,nl]);
pool = parallel.pool.PollableDataQueue;
spmd(nl)
    agent.l = labindex; % Agent index
    agent.na = length(I{agent.l});
    if use_congestion
        agent.Vl = costJ_base(I{agent.l})';
        agent.baseCost = costJ_base;
        agent.gl = g_congested(I{agent.l}, :, :); % Agent's knowledge of g
        %a.gl = g(I{a.l}, :, :).*(1/congestion(I{a.l}, :, :) - 1); % Agent's knowledge of g
    else
        agent.Vl = zeros(agent.na, 1);
        agent.baseCost = zeros(size(costJ));
        agent.gl = g(I{agent.l}, :, :); % Agent's knowledge of g
    end
    agent.Pl = P(I{agent.l}, :, :); % Agent's knowledge of P
    agent.I = I; % Agent's knowledge of global I
    agent.dl = D{agent.l};
    agent.r =  zeros(nl,1);%mean(a.gl,'all') * ones(nl,1);
    agent.r_old = zeros(nl,1);
    agent.pool = parallel.pool.PollableDataQueue;
end
toc

global agent_pools ...
        global_r global_epsilon global_iterations
global_r = zeros(nl,1);
global_epsilon = zeros(nl,1);
global_iterations = 0;
agent_pools = cell(nl,1);
for m = 1:nl
    a = agent{m};
    agent_pools{m} = a.pool;
end

max_iter = 50;
disp_on = true;
costJ_agg = nan(1,nx);
tic;
spmd(nl)
    iterations = 0;
    status = zeros(nl,1);
    while sum(status) < nl && iterations < max_iter
        %% Compute optimal next r
        agent.r_old(agent.l) = agent.r(agent.l);
        in_extra_args.baseCost = agent.baseCost;
        [agent.rl_new, agent.Vl, out_extra_args] = local_value_iteration(agent.r, agent.l, agent.I, agent.Vl, agent.gl, agent.Pl, alpha, agent.dl, in_extra_args);
        agent.r(agent.l) = agent.rl_new;    
        epsilon = agent.rl_new - agent.r_old(agent.l);
        
        % Send out new r
        send_msg = {agent.l, agent.r(agent.l)};
        send(message_pool, send_msg);

        % Receive message
        [receive_msg, OK] = poll(agent.pool, 1);
        while OK
            if isnan(receive_msg{2})
                status(receive_msg{1}) = 1;
            else
                agent.r_old(receive_msg{1}) = agent.r(receive_msg{1});
                agent.r(receive_msg{1}) = receive_msg{2};
            end
            [receive_msg, OK] = poll(agent.pool, 0.1);
        end

        % Check status of completion
        if max(abs(agent.r_old - agent.r)) < 1e-5 && iterations > 1
            status(agent.l) = 1;
            send_msg = {agent.l, nan};
            send(message_pool, send_msg);
        end

        iterations = iterations + 1;
    end
end

for l = 1:nl
    local_agent = agent{l};
    I{l} = local_agent.I{l};
    costJ_agg(I{l}) = local_agent.Vl;
end
   
toc
a1 = agent{1};
plot_cost(costJ, costJ_agg, a1.r, I);
disp('max error:')
disp(max(abs(costJ-costJ_agg)))
disp('average error')
disp(mean(abs(costJ-costJ_agg)))

%% afterEach Functions

function relay(msg)
    global agent_pools ...
        global_r global_epsilon global_iterations
    sender = msg{1};
    global_iterations = global_iterations + 1;
    for m = 1:length(agent_pools)
        if m == sender
            if ~isnan(msg{2})
                global_epsilon(m) = abs(global_r(m) - msg{2});
                global_r(m) = msg{2};
            end
            continue
        end
        send(agent_pools{m}, msg);
    end
    if mod(global_iterations, 10) == 0
        disp(['epsilon: ', num2str(global_epsilon')]);
    end
end