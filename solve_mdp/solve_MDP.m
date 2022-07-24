%% Discount factor.
alpha = 1;

convergence_tol = 1e-4;
compute_true = true;
%% Value Iteration
% Initialize variables that we update during value iteration.
% Cost (here it really is the reward):
if compute_true
    tic;
    [costJ, policy] = valueIteration(g, P, alpha, 1000, convergence_tol);
    toc
    
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

labelnode(graphPlot,1:length(costJ),costJ)
%%
I_init = I;
%% Aggregate Problem
clear agent
I = I_init;

p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    p = parpool('local', nl, 'AttachedFiles', 'local_value_iteration.m');
    %p = parpool('threads');
end

tic
agent = Composite([2,nl]);
for m =1:nl
    a.l = m; % Agent index
    a.na = length(I{a.l});
    a.Vl = ones(a.na, 1)*mean(costJ);
    a.gl = g(I{a.l}, :, :); % Agent's knowledge of g
    a.Pl = P(I{a.l}, :, :); % Agent's knowledge of P
    a.I = I; % Agent's knowledge of global I
    a.dl = D{a.l};
    a.r =  mean(a.gl,'all') * ones(nl,1);
    a.r_old = zeros(nl,1);
    agent{m} = a;
end
toc

%clear P g a

disp_on = true;
costJ_agg = nan(1,nx);
tic;
ticBytes(p);
spmd(nl)
    iterations = 0;
    mypartner = 1 + mod((labindex + 1) - 1, numlabs);
    status = zeros(nl,1);
    if agent.l == 1
        sendtoken = true;
    else
        sendtoken = false;
    end
    msg = {};
    while sum(status) < nl
        %% Compute optimal next r
        agent.r_old(agent.l) = agent.r(agent.l);
        in_extra_args.convergence_tol = max(1e-5,min(0.1, 10^(-10*iterations)));
        [agent.rl_new, agent.Vl, out_extra_args] = local_value_iteration(agent.r, agent.l, agent.I, agent.Vl, agent.gl, agent.Pl, alpha, agent.dl, in_extra_args);
        agent.r(agent.l) = agent.rl_new;    
        epsilon = agent.rl_new - agent.r_old(agent.l);
        %% Broadcast to all agents
        % Broadcast tags
        %   0 - msg
        %       msg = {[from, to, content], ...
        %              [from, to, content]}
        %               to = 0 : all agents 
        %   1 - r
        %   2 - I
        %   3 - full agent
        % Step 1 : Receive --------------------------------------------

        [tf, sources, tags] = labProbe('any');
        if tf
            for i = find(tags==0) % receive messages
                msg{end+1} = labReceive(sources(i), 0);
            end
            for i = find(tags==1) % receive r values
                agent.r_old(sources(i)) = agent.r(sources(i));
                agent.r(sources(i)) = labReceive(sources(i), 1);
                labSend(agent.rl_new, sources(i), 1);
            end
        end
     
        % evaluate messages
        mrk_fr_dlt = [];
        for i = 1:numel(msg)
            % check if message is for me
            if msg{i}{2} == agent.l || msg{i}{2} == 0 
                switch msg{i}{3}
                    case 'done'
                        % mark the sender as being done
                        status(msg{i}{1}) = 1; 
                    case 'sendtoken'
                        % set myself as the new sender
                        sendtoken = true;
                        iterations = iterations + 1;
                    case 'I'
                        labSend(agent.I{agent.l}, msg{i}{1}, 2);
                        agent.I{msg{i}{1}} = labReceive(msg{i}{1},2);
                end
                if msg{i}{2} == agent.l || msg{i}{2} == 0
                    % delete message once processed
                    mrk_fr_dlt(end+1) = i; 
                end
            end
        end
        msg(mrk_fr_dlt) = [];  
        % Step 2 : Send -----------------------------------------------
        if sendtoken
            if max(abs(agent.r_old - agent.r)) < 1e-5 && iterations > 2
                status(agent.l) = 1;
                if sum(status) == nl
                    sendtoken = 0;
                    for m = setdiff(1:numlabs, agent.l)
                        labSend({agent.l, 0, 'done'}, m, 0);
                    end
                    break
                else
                    for m = setdiff(1:numlabs, agent.l)
                        labSend({agent.l, 0, 'done'}, m, 0);
                    end
                end
            end
            % Pass on my r
            for m = setdiff(1:numlabs, agent.l)
                labSend(agent.rl_new, m, 1);
                agent.r_old(m) = agent.r(m);
                agent.r(m) = labReceive(m, 1);
            end
            if disp_on
                dispstr  = ['eps: ', num2str(max(abs(agent.r_old - agent.r)))];
                for l=1:nl
                    dispstr = [dispstr, '   r(',num2str(l),')=',num2str(agent.r(l),'%6.4f')];
                end
                disp(dispstr);
            end

            % Pass on token
            sendtoken = 0;
            while status(mypartner) && mypartner==labindex% update mypartner if they are done
                mypartner = mod(mypartner, numlabs) + 1;
            end
            labSend({agent.l, mypartner, 'sendtoken'}, mypartner, 0);
        end
    end
end
tocBytes(p)

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

figure;
G = simplify(G);
imshow(img); hold on;
graphPlot = plot(G,'XData',G.Nodes.XData,'YData',G.Nodes.YData, 'LineWidth',5, 'MarkerSize', 10);
highlight(graphPlot,{'DEST'},'NodeColor','g')

% labelnode(graphPlot,1:length(costJ_agg),costJ_agg)