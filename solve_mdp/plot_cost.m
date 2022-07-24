function plot_cost(costJ, costJ_agg, r, I)

    nx = length(costJ_agg);
    [I_lst, r_lst, agent_lst] = cell2list(I, r, nx);
    
    hold off;
    [cost, cost_index] = sort_single_col([costJ', costJ_agg', r_lst, agent_lst], 1);
    stem(1:nx, cost(:,1), 'filled'); hold on;
    stem(1:nx, cost(:,2), ':diamondr');

    nl = max(agent_lst);
    if nl > 8
        warning('plotting only supported for up to 6 aggregate states');
    end
    for i = 1:nx
        switch cost(i,4)
            case 1
                c = 'red';
            case 2
                c = 'green';
            case 3
                c = 'blue';	
            case 4
                c = 'cyan';	
            case 5
                c = 'magenta';	
            case 6
                c = 'yellow';	
            case 7
                c = '#D95319';		
            case 8
                c = 'black';		
            otherwise
                c = '#77AC30';
        end
        stem(i, cost(i,3), 'filled', 'LineStyle', 'none', 'Color', c, 'Marker', 'v');
    end
    legend('J^*', 'J_{agg}', 'r')
    axis = gca;
    axis.YLim = [min(costJ)*0.9, max(costJ)*1.1];

end



function [I_lst, r_lst, agent_lst] =cell2list(I, r, nx)

    I_lst = nan(1, nx);
    r_lst = nan(1, nx);
    agent_lst = nan(1, nx);
    c = 1;
    for m = 1:length(I)
        tmp = I{m};
        I_lst(c:(c+length(tmp)-1)) = tmp;
        r_lst(c:(c+length(tmp)-1)) = ones(1, length(tmp)) * r(m);
        agent_lst(c:(c+length(tmp)-1)) = ones(1, length(tmp)) * m;

        c = c + length(tmp);
    end
    lst = sort_single_col([I_lst', r_lst', agent_lst'], 1);
    I_lst = lst(:,1);
    r_lst = lst(:,2);
    agent_lst = lst(:,3);
end