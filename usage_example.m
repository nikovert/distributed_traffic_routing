% example script for converting a openstreetmap file to Graph and
% subsequently a Markov Chain
%
% use the example map.osm file in the release on github
%
% or
%
% download an OpenStreetMap XML Data file (extension .osm) from the
% OpenStreetMap website:
%   http://www.openstreetmap.org/
% after zooming in the area of interest and using the "Export" option to
% save it as an OpenStreetMap XML Data file, selecting this from the
% "Format to Export" options. The OSM XML is specified in:
%   http://wiki.openstreetmap.org/wiki/.osm
%
% See also PARSE_OPENSTREETMAP, PLOT_WAY, EXTRACT_CONNECTIVITY,
%          GET_UNIQUE_NODE_XY, ROUTE_PLANNER, PLOT_ROUTE, PLOT_NODES.
%
% derived from the openstreetmap add on by Ioannis Filippidis, jfilippidis@gmail.com

addpath('osm_parsing')
addpath('../Add-Ons/xml2struct')
%% name file
openstreetmap_filename = 'maps/oxford.osm';
%map_img_filename = 'map.png'; % image file saved from online, if available

%% convert XML -> MATLAB struct
% convert the OpenStreetMap XML Data file donwloaded as map.osm
% to a MATLAB structure containing part of the information describing the
% transportation network
%openstreetmap_filename = cleanfile(openstreetmap_filename);
[parsed_osm, osm_xml] = parse_openstreetmap(openstreetmap_filename);

%% find connectivity
parsed_osm = cleanways(parsed_osm);
[connectivity_matrix, intersection_node_indices] = extract_connectivity(parsed_osm);
intersection_nodes = get_unique_node_xy(parsed_osm, intersection_node_indices);
[~, node, way, ~] = assign_from_parsed(parsed_osm);
ID = node.id';
XData = node.xy(1,:)';
YData = node.xy(2,:)';
NodeTable = table(ID, XData, YData);
connectivity_matrix(end+1:length(node.id), end+1:length(node.id)) = zeros(length(node.id)-length(connectivity_matrix));
dG = digraph(connectivity_matrix, NodeTable); % directed graph

[bin,binsize] = conncomp(dG, 'Type', 'strong');
idx = binsize(bin) == max(binsize);
SG = subgraph(dG, idx);
p = plot(SG, 'XData',SG.Nodes.XData,'YData',SG.Nodes.YData);

%% Plot on map
figure(100);
gx = geoaxes;
geobasemap(gx,'streets');
gx.FontSize = 12;
%geolimits([min(SG.Nodes.YData) max(SG.Nodes.YData)],[min(SG.Nodes.XData) max(SG.Nodes.XData)])
geolimits(gx, [51.7367   51.7520],[-1.2483   -1.2185])
hold on
geoscatter(gx, SG.Nodes.YData, SG.Nodes.XData)

from_y = SG.Nodes.YData(SG.Edges.EndNodes(:,1));
from_x = SG.Nodes.XData(SG.Edges.EndNodes(:,1));
to_y = SG.Nodes.YData(SG.Edges.EndNodes(:,2));
to_x = SG.Nodes.XData(SG.Edges.EndNodes(:,2));
for e = 1:size(SG.Edges,1)
    geoplot(gx, [from_y(e) to_y(e)],[from_x(e) to_x(e)], 'b:');
end
%% plan a route

% try with the assumption of one-way roads (ways in OSM)
% start = 2164; % node global index
% target = 1196;
% [route, dist] = shortestpath(SG, start, target,'Method','unweighted');
% highlight(p,route,'EdgeColor','r')

%% Create destination node
G = SG;
dest_node = 4;
destination_ID = 1e+10;
NodeProps = table(destination_ID, G.Nodes.XData(2626), G.Nodes.YData(4), ...
    'VariableNames', {'ID' 'XData', 'YData'});
G = addnode(G, NodeProps);
G = addedge(G, dest_node, length(G.Nodes.ID), 1);
figure(10)
graphPlot = plot(G, 'XData',G.Nodes.XData,'YData',G.Nodes.YData);
xlabel('Longitude', 'FontSize',15, 'Interpreter','latex')
ylabel('Latitude', 'FontSize',15, 'Interpreter','latex')
%% Create Regions
clustercount = 5;
use_costAggregation = false;
if exist('costJ_base') && use_costAggregation
    data = costJ_base';
else
    data = [G.Nodes.XData, G.Nodes.YData];
end
[obj, c, ~, D] = kmeans(data,clustercount, 'Distance','sqeuclidean');

G.Nodes.Cluster = obj;
%% Create convex hulls
I = cell(clustercount,1);
figure(10)
hold on;
for ind = 1:clustercount
    I{ind} = find(obj==ind)';
    x = G.Nodes.XData(I{ind});
    y = G.Nodes.YData(I{ind});
    if length(I{ind}) > 2
        k = convhull(x,y);
        fill(x(k), y(k), 'r','facealpha', 0.5 );
    end
end

%%
nu = max(outdegree(G));
nx = height(G.Nodes);
nl = clustercount;
dest_index = length(G.Nodes.ID);

% Create P(i,u,j)
inf_value = 10000;
P = zeros(nx,nx,nu);
g = ones(nx,nx,nu)*inf_value;
% WARNING THIS MIGHT LEAD TO zero prob transition being favoured
for i = 1:nx
    if i == dest_index
        g(i, :, :) = 0;
        P(i, i, 1) = 0;
        continue
    end
    N = successors(G,i);
    for j_index=1:outdegree(G,i)
        P(i, N(j_index), j_index) = 1;
        idxOut = findedge(G,i,N(j_index));

        if N(j_index) == dest_index
            g(i, N(j_index), j_index) = 0;
        else
            % set the cost to go, based on the distance and speed limit
            arclen = distance('gc',...
                [G.Nodes.XData(i),G.Nodes.YData(i)], ...
                [G.Nodes.XData(N(j_index)),G.Nodes.YData(N(j_index))]);
            g(i, N(j_index), j_index) = 60*deg2sm(arclen)/G.Edges.Weight(idxOut); % 60 to convert from hours to minutes
        end
    end
    P(i, i, outdegree(G,i)+1:nu) = 1;
    g(i, i, outdegree(G,i)+1:nu) = inf_value;

end
assert(min((sum(P, 2) - 1) < 10e-5, [], 'all'))

phi = @(i,l) ismember(i, I{l});

D = cell(nl,1);
for i = 1:nl
    % compute dissagregation probability
    total_in = 0;
    index = 0;
    incoming_w = zeros(1,length(I{i}));
    for node = I{i}
        index = index + 1;

        % Compute ingoing edges to calculate dissagregation prob
        [eid,nid] = outedges(G, node); % if directed, this needs to be inedges
        nid_new = setdiff(nid,I{i});
        [~,~,IB] = intersect(nid_new, nid);
        %incoming_w(index) = sum(G.Edges.Weight(eid(IB)));
        incoming_w(index) = ~isempty(eid(IB)); %length(eid(IB));
        total_in = total_in + incoming_w(index);
    end
    D{i} = incoming_w/total_in;

%     Alternative even D
%     D{i} = ones(size(I{i}))/length(I{i});
end
%% Solve congested MDP problem
addpath('solve_mdp')
solve_congestedMDP_threadedv2
