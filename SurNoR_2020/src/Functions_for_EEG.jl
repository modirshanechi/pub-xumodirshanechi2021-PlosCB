# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Plotting correlations
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_corr_plot_mat(X;Path="untitled",Color = "#B71616",y_lim=[-0.2,0.35],
                              Title="Average correlation with EEG amplitude",
                              X_ticks=25:50:500,p_plot=0,p_thresh=0.05,
                              y_lim_p=[-10,0],x_lim=[-100,650])
    time = X[1,:]
    y = X[2,:]
    dy = X[3,:]
    p_obs = X[4,:]
    S = X[5,:]

    fig = figure(figsize=(7,4)); ax = gca()
    ax.plot(time,zeros(size(y)),linestyle="dashed",linewidth=0.5,color="k")
    ax.plot([0,0],[-10,10],linestyle="dashed",linewidth=0.5,color="k")

    if sum(S .== 1)>0
        # t_min = findmin(time[S .== 1])[1]
        # t_max = findmax(time[S .== 1])[1]
        # t_S = [t_min-25,t_max+25]

        t_S1 = time[2:end]
        t_S1 = t_S1[(S[2:end] - S[1:(end-1)]).==1]
        t_S2 = time[1:(end-1)]
        t_S2 = t_S2[(S[2:end] - S[1:(end-1)]).==-1]

        for i=1:length(t_S1)
            t_S = [t_S1[i]-5,t_S2[i]+5]
            ax.fill_between(t_S,zeros(size(t_S)).- 10, 20 .* ones(size(t_S)) .- 10,
                            color="#464747",alpha=0.05,linewidth=0)
        end
    end

    ax.plot(time,y,color=Color)
    ax.fill_between(time,y-dy,y+dy,color=Color,alpha=0.1,linewidth=0)

    title(Title)
    xlabel("time [ms]")
    ax.set_xticks(X_ticks)
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)
    savefig(string(Path,".pdf"))
    savefig(string(Path,".svg"))
    close(fig)

    if p_plot==1
        fig = figure(figsize=(7,4)); ax = gca()
        ax.plot(time,ones(size(y)) .* log(p_thresh),linestyle="dashed",linewidth=0.5,color="k")
        ax.plot([0,0],[-10,10],linestyle="dashed",linewidth=0.5,color="k")

        if sum(S .== 1)>0
            t_S1 = time[2:end]
            t_S1 = t_S1[(S[2:end] - S[1:(end-1)]).==1]
            t_S2 = time[1:(end-1)]
            t_S2 = t_S2[(S[2:end] - S[1:(end-1)]).==-1]

            for i=1:length(t_S1)
                t_S = [t_S1[i]-5,t_S2[i]+5]
                ax.fill_between(t_S,zeros(size(t_S)).- 100, 100 .* ones(size(t_S)),
                                color="#464747",alpha=0.05,linewidth=0)
            end
        end

        ax.plot(time,log.(p_obs),color="#B2151D")

        title(string(Title, " - P values"))
        xlabel("time [ms]")
        ax.set_xticks(X_ticks)
        ax.set_ylim(y_lim_p)
        ax.set_xlim(x_lim)
        savefig(string(Path,"_pval",".pdf"))
        savefig(string(Path,"_pval",".svg"))
        close(fig)
    end
end
