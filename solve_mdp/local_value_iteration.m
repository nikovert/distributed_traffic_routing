function [rl_new, V_next, out_extra_args] = local_value_iteration(r, l, I, Vl, gl, Pl, alpha, d, in_extra_args)
    
    nx = size(gl,2);
    nl = size(I,1);
    nu = size(Pl, 3);
    
    if nargin < 9
        in_extra_args = [];
    end

    if isfield(in_extra_args, 'e')
        e = in_extra_args.e;
    end
    if isfield(in_extra_args, 'convergence_tol')
        convergence_tol = in_extra_args.convergence_tol;
    else
        convergence_tol = 1e-4;
    end

    if isfield(in_extra_args, 'baseCost')
        baseCost = in_extra_args.baseCost;
    else
        baseCost = zeros(1,nx);
    end

    if isnumeric(I)
        I = mat2cell(I,ones(1,nl),nx/nl);
    end

    assert(size(I{l},2) == size(gl, 1))
    assert(size(I{l},2) == size(Pl, 1))

    cost_to_go = zeros(1,nx);
    cost_to_go_min = zeros(1,nx);
    for m = 1:nl
        if m == l
            cost_to_go(I{l}) = Vl;
        else
            cost_to_go(I{m}) = r(m) + baseCost(I{m});
        end
    end
    %% For sanity checks
        Vl_old = Vl;
    %%
    % compute updated for agents bias function Vl
    
    u_opt_index = zeros(size(I{l}));  
    delta = inf(size(I{l}));
    while max(abs(delta)) > abs(convergence_tol)
        Vl_index = 1;
        for i = I{l}  
                % Vl(Vl_index)  is a short expression of the following
                %         V_tmp = 0;
                %         for j = 1:nx
                %             V_tmp = V_tmp + P(i,j,:) .* (double(g(i,j,:)) + alpha * cost_to_go(j));
                %         end
                %         min(V_tmp, [], 3)- rl(l);
            [cost_tmp, u_opt_index(Vl_index)] = min( squeeze(sum(double(gl(Vl_index,:,:)) .* Pl(Vl_index,:,:), 2))' + alpha* cost_to_go     * squeeze(Pl(Vl_index,:,:)) );
            
            Vl(Vl_index) = cost_tmp;
            delta(Vl_index) = Vl(Vl_index) - cost_to_go(i);
            
            cost_to_go(i) = Vl(Vl_index); 

            Vl_index = Vl_index + 1;
        end
    end

    % Find agents fixed point r_l
    if ~isempty(Vl)
        assert(size(Vl',2) == size(I{l},2))
    end
    
%     dlj = zeros(numel(I{l}), numel(I{l}), nu); 
%     dlj_index = 1;
%     for j = I{l}
%         % take max over all i
%         for h = 1:length(I{l})
% %             p_sum = 0;
% %             for i = 1:nx
% %                 if ~ismember(i, I{l})
% %                     p_sum = p_sum + Pl(h,i,u_opt_index(h)); 
% %                 end
% %             end
%             for u = 1:nu
%                 dlj(dlj_index, h, u) = max(Pl(h,j,u) + sum(Pl(h, setdiff(1:nx, I{l}), u))/numel(I{l}));  
%             end
%         end
%         dlj_index = dlj_index + 1;
%     end
    if isnumeric(d)
        rl_new = d * (Vl - baseCost(I{l})'); 
    else
        rl_new = feval(d, I{l}, l) * (Vl - baseCost(I{l})'); 
    end
    out_extra_args.el_new = max(abs(rl_new - (Vl - baseCost(I{l})')));
%     rl_new = 0;
%     for h = 1:length(I{l})
%         for u = 1:nu
%             rl_tmp = dlj(:,h,u)' * (Vl); 
%             if rl_new <= rl_tmp
%                 rl_new = rl_tmp;
%             end
%         end
%     end
    %% For sanity checks
        %delta = Vl - Vl_old;
        %eps = rl_new - r(l);
%         assert(max(eps) <= max(delta))
    %%
    % update agents global bias function with Vl
    if nargout > 1
        %V_next = V_tmp(I{l});
        V_next = Vl;
    end
end