# The code to generate S5 Fig
using PyPlot
using SurNoR_2020
using Statistics
PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# WARNING: put n_set = 1 for the left panel, put n_set = 2 for the middle,
#          and put n_set = 3 for the right panel
n_set = 1
# ------------------------------------------------------------------------------
# Training Algorithms
# ------------------------------------------------------------------------------
Agents_SurNoR = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(3)
Model = Agents_SurNoR[[2,3,4,5,6,7,8,9,10,12],:,:]

Param_Path = string("data/Fitted_parameters_simulated/Overall/nset_", string(n_set), "/")
Agents_SurNoR_nset   = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(1,Param_Path = Param_Path)
Model_nset = Agents_SurNoR_nset[[2,3,4,5,6,7,8,9,10,12],:,:]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Correlation between model variables given the parameters fitted to the behavior
# of participants and the behavior of the simulated participants
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Putting all RPE, NPE, Novelty, and Surprise together
Sub_Set = 1:10

Block_Set = 1:2
Epi_Set = 1:5

Concat_Signals = []
Concat_Signals_nset = []
for Sub=Sub_Set
    for Block_ind = 1:length(Block_Set)
        Block = Block_Set[Block_ind]
        for Epi_ind = 1:length(Epi_Set)
            Epi = Epi_Set[Epi_ind]
            temp = Model[Sub,Block,Epi]
            temp_nset = Model_nset[Sub,Block,Epi]
            if (Epi_ind==1)&(Block_ind==1)
                Sub_Signal = cat(temp.Surprise,temp.Novelty_Seq,temp.δ_N_Seq,temp.δ_R_Seq,temp.Input.obs.==0,dims=2)
                Sub_Signal_nset = cat(temp_nset.Surprise,temp_nset.Novelty_Seq,temp_nset.δ_N_Seq,temp_nset.δ_R_Seq,temp_nset.Input.obs.==0,dims=2)
            else
                Sub_Signal = cat(Sub_Signal,
                                 cat(temp.Surprise,temp.Novelty_Seq,temp.δ_N_Seq,temp.δ_R_Seq,temp.Input.obs.==0,dims=2),
                                 dims=1)
                Sub_Signal_nset = cat(Sub_Signal_nset,
                                 cat(temp_nset.Surprise,temp_nset.Novelty_Seq,temp_nset.δ_N_Seq,temp_nset.δ_R_Seq,temp_nset.Input.obs.==0,dims=2),
                                 dims=1)
            end
        end
    end
    Sub_Signal = Sub_Signal[Sub_Signal[:,1] .!= -1,:]
    append!(Concat_Signals,[deepcopy(Sub_Signal)])
    Sub_Signal_nset = Sub_Signal_nset[Sub_Signal_nset[:,1] .!= -1,:]
    append!(Concat_Signals_nset,[deepcopy(Sub_Signal_nset)])
end


# Barplot
Signal_names = ["Surprise","Novelty","NPE","RPE","Goal"]
N_Sig = length(Signal_names)
N_Sub = length(Sub_Set)
Corr_Mat = zeros(N_Sub,N_Sig)
for Sub=1:N_Sub
    for Sig = 1:N_Sig
        Corr_Mat[Sub,Sig] = cor(Concat_Signals[Sub][:,Sig],Concat_Signals_nset[Sub][:,Sig])
    end
end

Colors = ["#B159E0","#3950A3","#02B0FB","#391748","#B21F24"]

fig = figure(); ax = gca()
x = 1:N_Sig

y = mean(Corr_Mat,dims=1)
dy = std(Corr_Mat,dims=1)/sqrt(N_Sub)
Sig_inds = [1,3,4]
ax.plot([x[1]-0.5,x[length(Sig_inds)]+0.5],ones(2) .* 1, "--k",alpha=0.5)
ax.plot([x[1]-0.5,x[length(Sig_inds)]+0.5],ones(2) .* 0.95, "--k",alpha=0.5)
for i = 1:length(Sig_inds)
    ax.bar(i, y[Sig_inds[i]] , yerr=dy[Sig_inds[i]], color=Colors[Sig_inds[i]])
    ax.errorbar(i, y[Sig_inds[i]] , yerr=dy[Sig_inds[i]],color="k",linewidth=1,capsize=5,linestyle ="")
end

σ = 0.1
for j = 1:N_Sub
  x_1 = (1:length(Sig_inds)) .+ 2*σ*(rand() - 0.5)
  y = Corr_Mat[j,Sig_inds]
  ax.plot(x_1, y[:],".",color="k",alpha=0.1)
end
ax.set_ylim([0.8,1.02])
ax.set_xlim([x[1]-0.5,x[length(Sig_inds)]+0.5])

ax.set_xticks(1:length(Sig_inds))
ax.set_xticklabels(Signal_names[Sig_inds])
