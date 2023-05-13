function plot_cost(costJ, costJ_agg, r, I)
    nx = length(costJ_agg);
    [I_lst, r_lst, agent_lst] = cell2list(I, r, nx);
    
    hold off;
    [cost, cost_index] = sort_single_col([costJ', costJ_agg', r_lst, agent_lst], 1);
    stairs(1:nx, cost(:,2), ':diamondr', 'MarkerSize', 1);hold on;
    scatter(1:nx, cost(:,1),1,'MarkerFaceColor','b','MarkerEdgeColor','b',...
    'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); 
    
    legend('J_{agg}', 'J^*')

%     nl = max(agent_lst);
%     if nl > 8
%         warning('plotting only supported for up to 6 aggregate states');
%     end
%     for i = 1:nx
%         switch cost(i,4)
%             case 1
%                 c = 'red';
%             case 2
%                 c = 'green';
%             case 3
%                 c = 'blue';	
%             case 4
%                 c = 'cyan';	
%             case 5
%                 c = 'magenta';	
%             case 6
%                 c = 'yellow';	
%             case 7
%                 c = '#D95319';		
%             case 8
%                 c = 'black';		
%             otherwise
%                 c = '#77AC30';
%         end
%         stem(i, cost(i,3), 'LineStyle', 'none', 'Color', c, 'Marker', 'v');
%     end
%     switch cost(i,4)
%         case 1
%             legend('J^*', 'J_{agg}', 'r_1')
%         case 2
%             legend('J^*', 'J_{agg}', 'r_1', 'r_2')
%         case 3
%             legend('J^*', 'J_{agg}', 'r_1', 'r_2', 'r_3')
%         case 4
%             legend('J^*', 'J_{agg}', 'r_1', 'r_2', 'r_3', 'r_4')
%         case 5
%             legend('J^*', 'J_{agg}', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5')
%         case 6
%             legend('J^*', 'J_{agg}', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6')
%         case 7
%             legend('J^*', 'J_{agg}', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7')
%         case 8
%             legend('J^*', 'J_{agg}', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7', 'r_8')	
%         otherwise
%             legend('J^*', 'J_{agg}', 'r')
%     end
    axis = gca;
    axis.YLim = [min(min(r),min(costJ)*0.9), max(costJ)*1.1];
    axis.XLim = [1, size(cost,1)];
    % Create ylabel
    ylabel('Cost J','Interpreter','latex');
    
    % Create xlabel
    xlabel('Node number (sorted by $J^*$)','Interpreter','latex');

    % Set the remaining axes properties
    set(axis,'FontSize',15, 'TickLabelInterpreter','latex');
    % Create legend
    legend1 = legend(axis,'show');
    set(legend1,...
        'Position',[0.160510716851965 0.780555568670876 0.101231189212786 0.108333330424059]);
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