function graphPlot = plot_optimalEdge(G, policy)
% plots the optimal policy route over the graph G
    graphPlot = plot(G, 'XData',G.Nodes.XData,'YData',G.Nodes.YData);
    for node = 1:length(G.Nodes.ID)-1
        N = successors(G,node);
        u = policy(node);
        highlight(graphPlot,node,N(u),'EdgeColor','red');
    end
end