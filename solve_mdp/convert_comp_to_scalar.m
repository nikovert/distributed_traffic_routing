function [varargout] = convert_comp_to_scalar(varargin)
    varargout = cell(1, nargin);
    for index = 1:nargin
        variable = varargin{index};
        if isnumeric(variable) || isa(variable,'function_handle')
            varargout{index} = variable;
        else
            varargout{index} = variable{1};
        end
    end
end