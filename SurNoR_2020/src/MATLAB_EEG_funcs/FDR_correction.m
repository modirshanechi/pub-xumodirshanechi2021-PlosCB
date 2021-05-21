function Output = FDR_correction(X,q)
    % This function uses the algorithm of Benjamini and Hochberg for False
    % Discovery Rate control
    % Output = FDR_correction(X,q)
    %   X = n_sub * n_var
    %   q = the bound for FDR
    %   corresponding paper: 
    %           https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
    T = size(X,2);
    p_val_obs = zeros(1,T);
    tstat_obs = zeros(1,T);
    for t=1:T
        %[~,p] = ttest(X(:,t));
        [~,p,~,stats] = ttest(X(:,t));
        p_val_obs(t) = p;
        tstat_obs(t) = stats.tstat;
    end
    
    df_obs = stats.df;

    p_sort = sort(p_val_obs);
    FDR_line = (1:T)/T*q;
    ind = 1:T;
    FDR_ind = max([1,ind(p_sort<=FDR_line)]);
    p_thresh = FDR_line(FDR_ind);
    Sign_t = p_val_obs<p_thresh;
    
    Output = struct();
    Output.Sign_t = Sign_t;
    Output.p_val_obs = p_val_obs;
    Output.p_thresh = p_thresh;
    Output.tstat_obs = tstat_obs;
    Output.df_obs = df_obs;
end
