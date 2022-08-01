%% Discount factor.
convergence_tol = 1e-6;
compute_true = true;
use_congestion = true;
alpha = 0.99;
%% Value Iteration
% Initialize variables that we update during value iteration.
% Cost (here it really is the reward):
if compute_true
    tic;
    [costJ_base, policy_base] = valueIteration(g, P, alpha, 10000, convergence_tol);
    graphPlot = plot_optimalEdge(G, policy_base);
    toc
    
    if use_congestion
        alpha = 0.9;
        congestion = min(1, 0.25 + rand(size(g)));
        g_congested = g ./congestion;
    
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

%% Aggregate Problem
clear agentList in_extra_args

p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    p = parpool('threads');
else
    delete(gcp)
    p = parpool('threads');
end


message_pool = parallel.pool.DataQueue;
afterEach(message_pool, @relay);

tic
agentList = cell(nl,1);
use_baseCost = false;
for m = 1:nl
    agent.l = m; % Agent index
    agent.na = length(I{agent.l});
    if use_congestion
        agent.gl = g_congested(I{agent.l}, :, :); % Agent's knowledge of g
        %agent.gl = g(I{agent.l}, :, :).*(1/congestion(I{agent.l}, :, :) - 1); % Agent's knowledge of g
    else
        agent.gl = g(I{agent.l}, :, :); % Agent's knowledge of g
    end

    if use_congestion && use_baseCost
        agent.Vl = costJ_base(I{agent.l})';
        agent.baseCost = costJ_base;
    else
        agent.Vl = zeros(agent.na, 1);
        agent.baseCost = zeros(size(costJ));
    end
    agent.Pl = P(I{agent.l}, :, :); % Agent's knowledge of P
    agent.I = I; % Agent's knowledge of global I
    agent.dl = D{agent.l};
    if any(size(c) ~= [nl,1])
        agent.r =  zeros(nl,1);
    else
        agent.r =  c;
    end
    agent.r_old = zeros(nl,1);
    agent.transmission_thresh = 1e-5;
    agentList{m} = agent;
end
toc

global agent_pools ...
        global_r global_epsilon global_iterations ...
        communication_order
global_r = zeros(nl,1);
global_r_prev = ones(nl,1);
global_epsilon = zeros(nl,1);
global_iterations = 0;
agent_pools = cell(nl,1);
communication_order = [];

transmission_history = cell(nl,1);
total_iterations = 0;
costJ_agg = nan(1,nx);
tic;
while max(abs(global_r_prev - global_r)) > 1.0e-02 || total_iterations == 0
    disp('starting a new iteration...')
    global_r_prev = global_r;
    parfor m = 1:nl
        [agentList{m}, iterations, t_hist] = solve_aggregatedValueIteration(agentList{m}, alpha, message_pool);
        transmission_history{m} = [transmission_history{m} t_hist];
    end
%     for m = 1:nl
%         agentList{m}.r(setdiff(1:nl, m)) = global_r(setdiff(1:nl, m));
%     end
    disp(['global_r: ', num2str(global_r')]);
    total_iterations = total_iterations + 1;
    break
end

delta = nan(nl,1);
for l = 1:nl
    costJ_agg(I{l}) = agentList{l}.Vl;
    delta(l) = max(agentList{l}.Vl) - min(agentList{l}.Vl);
end
   
tstart = inf;
for m=1:nl
    tstart = min(tstart, agentList{m}.rl_hist(2,1));
end

% Create figure
figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
for m=1:nl
    plot(agentList{m}.rl_hist(2,:)-tstart, agentList{m}.rl_hist(1,:),'DisplayName',['$r_', num2str(m), '$'])
    r_inter = interp1(agentList{m}.rl_hist(2,:),agentList{m}.rl_hist(1,:),transmission_history{m});
    scatter(transmission_history{m}-tstart, r_inter, 'DisplayName',['transmission of $r_', num2str(m), '$']);
end
% Create ylabel and xlabel
ylabel('r','Interpreter','latex');
xlabel({'time'},'Interpreter','latex');
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.630153488465238 0.135317454356997 0.257941786016342 0.363095230147952],...
    'Interpreter','latex');

toc
figure();
plot_cost(costJ, costJ_agg, global_r, I);
disp('max error:')
disp(max(abs(costJ-costJ_agg)))
disp('average error')
disp(mean(abs(costJ-costJ_agg)))

%% functions
function [agent, iterations, transmission_history] = solve_aggregatedValueIteration(agent, alpha, message_pool)
    nl = length(agent.r);
    max_iter = nl*10;
    iterations = 0;
    status = zeros(nl,1);
    
    % Create and send pool
    agent.pool = parallel.pool.PollableDataQueue;
    send_msg = {agent.l, inf, agent.pool};
    send(message_pool, send_msg);

    if ~isfield(agent, 'rl_hist')
        agent.rl_hist = [agent.r(agent.l); cputime];
    end
    transmission_history = [];
    last_transmitted_r = -inf;
    while sum(status) < nl && iterations < max_iter
        %% Compute optimal next r
        agent.r_old(agent.l) = agent.r(agent.l);
        in_extra_args.baseCost = agent.baseCost;
        [agent.rl_new, agent.Vl, out_extra_args] = local_value_iteration(agent.r, agent.l, agent.I, agent.Vl, agent.gl, agent.Pl, alpha, agent.dl, in_extra_args);
        agent.r(agent.l) = agent.rl_new;    
        agent.rl_hist(:, end+1) = [agent.r(agent.l); cputime];

        % Send out new r
        if abs(last_transmitted_r - agent.r(agent.l)) > agent.transmission_thresh
            send_msg = {agent.l, agent.r(agent.l)};
            transmission_history = [transmission_history cputime];
            send(message_pool, send_msg);
            last_transmitted_r = agent.r(agent.l);
        end

        % Receive message
        [receive_msg, OK] = poll(agent.pool, 1);
        while OK
            if isnan(receive_msg{2})
                status(receive_msg{1}) = 1;
            else
                agent.r_old(receive_msg{1}) = agent.r(receive_msg{1});
                agent.r(receive_msg{1}) = receive_msg{2};
            end
            [receive_msg, OK] = poll(agent.pool, 0.01);
        end

        % Check status of completion
        if max(abs(agent.r_old - agent.r)) < 1e-3 && iterations > 1
            status(agent.l) = 1;
            send_msg = {agent.l, nan};
            send(message_pool, send_msg);
        end

        iterations = iterations + 1;
    end

end

function relay(msg)
    global agent_pools ...
        global_r global_epsilon global_iterations ...
        communication_order
    sender = msg{1};
    global_iterations = global_iterations + 1;

    communication_order = [communication_order sender];

    if isinf(msg{2})
        agent_pools{sender} = msg{3};
        return
    end

    for m = 1:length(agent_pools)
        if m == sender
            if ~isnan(msg{2})
                global_epsilon(m) = abs(global_r(m) - msg{2});
                global_r(m) = msg{2};
            end
            continue
        end
        if ~isempty(agent_pools{m})
            send(agent_pools{m}, msg);
        end
    end
%     if mod(global_iterations, 100) == 0
%         disp(['epsilon: ', num2str(global_epsilon')]);
%     end
end