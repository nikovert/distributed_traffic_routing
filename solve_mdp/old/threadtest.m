%% Mini thread test
p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    p = parpool('threads');
end
message_pool = parallel.pool.DataQueue;
afterEach(message_pool, @relay);

nl = 3;
agent = Composite([2,nl]);
spmd(nl)
    agent.r = zeros(nl,1);
    agent.r_old = zeros(nl,1);
    agent.l = labindex;
    agent.pool = parallel.pool.PollableDataQueue;
end

global agent_pools
agent_pools = cell(nl,1);
for m = 1:nl
    a = agent{m};
    agent_pools{m} = a.pool;
end

spmd(nl)
    iterations = 0;
    max_iter = 100;
    status = zeros(nl,1);
    while sum(status) < nl && iterations < max_iter
        % Calculate new r
        agent.r_old(agent.l) = agent.r(agent.l);
        agent.r(agent.l) = agent.r(agent.l) + agent.l*10^iterations;
        epsilon = agent.r(agent.l) - agent.r_old(agent.l);

        % Send out new r
        send_msg = {agent.l, agent.r(agent.l)};
        send(message_pool, send_msg);

        % Receive message
        [receive_msg, OK] = poll(agent.pool, 1);
        while OK
            if isnan(receive_msg{2})
                status(receive_msg{1}) = 1;
            else
                agent.r(receive_msg{1}) = receive_msg{2};
            end
            [receive_msg, OK] = poll(agent.pool, 0.1);
        end

        iterations = iterations+1;
        if iterations > 5
            status(agent.l) = 1;
            send_msg = {agent.l, nan};
            send(message_pool, send_msg);
        end
    end
end

function relay(msg)
    global agent_pools
    sender = msg{1};
    for m = 1:length(agent_pools)
        if m == sender
            continue
        end
        send(agent_pools{m}, msg);
    end
end
