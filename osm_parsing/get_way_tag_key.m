function [key, val] = get_way_tag_key(tag, str)
% get tags and key values for ways
%
% 2010.11.21 (c) Ioannis Filippidis, jfilippidis@gmail.com
%
% See also PLOT_WAY, EXTRACT_CONNECTIVITY.
if nargin < 2
    str = 'highway';
end

if isstruct(tag) == 1
    key = tag.Attributes.k;
    val = tag.Attributes.v;
elseif iscell(tag) == 1
    for i=1:length(tag)
        key = tag{i}.Attributes.k;
        if strcmp(key, str)
            val = tag{i}.Attributes.v;
            return
        end
    end
    val = tag{i}.Attributes.v;
%       val = tag{1}.Attributes.v;
%       key = tag{1}.Attributes.k;
else
%     if isempty(tag)
%         warning('Way has NO tag.')
%     else
%         warning('Way has tag which is not a structure nor cell array, but:')
%         disp(tag)
%     end
    
    key = '';
    val = '';
end
