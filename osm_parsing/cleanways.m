function parsed_osm = cleanways(parsed_osm)
    delete_list = [];
    for w = 1:length(parsed_osm.way.id)
        tags = parsed_osm.way.tag{w};
        keepway = 0;
        for t = 1:length(tags)
            if length(tags)>1 && strcmp(tags{t}.Attributes.k, 'highway')
                keepway = keepway+1;
                continue
            end
            if length(tags)>1 && strcmp(tags{t}.Attributes.k, 'maxspeed')
                keepway = keepway+1;
                continue
            end
        end
        if keepway<2
            delete_list = [delete_list w];
        end
    end
    parsed_osm.way.id(delete_list) = [];
    parsed_osm.way.nd(delete_list) = [];
    parsed_osm.way.tag(delete_list) = [];
end