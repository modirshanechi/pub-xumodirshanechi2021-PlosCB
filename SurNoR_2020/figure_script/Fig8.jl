# The code to generate Fig 8
using PyPlot
using SurNoR_2020
using Statistics
using MAT
using HypothesisTests

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# WARNING: put n_set = 1 for the left panel, put n_set = 2 for the middle,
#          and put n_set = 3 for the right panel
n_set = 3
# ------------------------------------------------------------------------------
# Load simulated data
# ------------------------------------------------------------------------------
Sim_Data = load("data/Simulated_data/Simulated_data.jld")
Inputs = Sim_Data["Inputs"]

Sub_Num = 12
N_start = (n_set-1)*Sub_Num
BehavData = Inputs[N_start .+ (1:Sub_Num),:,:]

# ------------------------------------------------------------------------------
# Training Algorithms
# ------------------------------------------------------------------------------
Param_Path = string("data/Fitted_parameters_simulated/CV_3Fold/nset_", string(n_set), "/")
Fold_Set = [[9,10,11,12],[5,6,7,8],[1,2,3,4]]
for Part_ind = 1:4
    Agents_MFNS    = SurNoR_2020.Func_MFNS_CV_train(BehavData=BehavData,
                                                    Param_Path=Param_Path,
                                                    Fold_Set=Fold_Set,
                                                    Part_ind=Part_ind)
    Agents_HybSExp = SurNoR_2020.Func_HybSExp_CV_train(BehavData=BehavData,
                                                       Param_Path=Param_Path,
                                                       Fold_Set=Fold_Set,
                                                       Part_ind=Part_ind)
    Agents_HybN    = SurNoR_2020.Func_HybN_CV_train(BehavData=BehavData,
                                                    Param_Path=Param_Path,
                                                    Fold_Set=Fold_Set,
                                                    Part_ind=Part_ind)
    Agents_SurNoR  = SurNoR_2020.Func_SurNoR_CV_train(BehavData=BehavData,
                                                      Param_Path=Param_Path,
                                                      Fold_Set=Fold_Set,
                                                      Part_ind=Part_ind)

    if Part_ind == 1
        global Agents_Tot_Alg = cat(Agents_MFNS,Agents_HybSExp, Agents_HybN, Agents_SurNoR, dims=4)

    else
        Agents_Tot_Alg_temp = cat(Agents_MFNS,Agents_HybSExp, Agents_HybN, Agents_SurNoR, dims=4)

        Agents_Tot_Alg = cat(Agents_Tot_Alg, Agents_Tot_Alg_temp, dims=5)
    end
end
SurNoR_ind = 4   # index corresponding to SurNoR

# Extracting Log-likelihoods
Alg_Num = size(Agents_Tot_Alg)[4]
Part_Num = size(Agents_Tot_Alg)[5]
LogL_Matrix = zeros((Sub_Num,5,2,Alg_Num,Part_Num))
LogL_Unif_Matrix = zeros((Sub_Num,5,2,Alg_Num,Part_Num))

for Sub = 1:Sub_Num
    for Epi = 1:5
        for Block = 1:2
            for Alg = 1:Alg_Num
                for Part_ind = 1:Part_Num
                    LogL_Matrix[Sub,Epi,Block,Alg,Part_ind] =
                            Agents_Tot_Alg[Sub,Block,Epi,Alg,Part_ind].LogL
                    LogL_Unif_Matrix[Sub,Epi,Block,Alg,Part_ind] =
                            Agents_Tot_Alg[Sub,Block,Epi,Alg,Part_ind].LogL_Unif
                end
            end
        end
    end
end

LogL_Matrix_Epi1 = LogL_Matrix[:,1,1,:,:]
LogL_Unif_Matrix_Epi1 = LogL_Unif_Matrix[:,1,1,:,:]

LogL_Matrix_Epi1B2 = LogL_Matrix[:,1,2,:,:]
LogL_Unif_Matrix_Epi1B2 = LogL_Unif_Matrix[:,1,2,:,:]

LogL_Sum = sum(LogL_Matrix,dims=1)
LogL_Unif_Sum = sum(LogL_Unif_Matrix,dims=1)

LogL_Sum_Epi1 = sum(LogL_Matrix_Epi1,dims=1)
LogL_Unif_Sum_Epi1 = sum(LogL_Unif_Matrix_Epi1,dims=1)

LogL_Sum_Epi1B2 = sum(LogL_Matrix_Epi1B2,dims=1)
LogL_Unif_Sum_Epi1B2 = sum(LogL_Unif_Matrix_Epi1B2,dims=1)

dLogL_Sum_Block = sum(LogL_Sum - LogL_Unif_Sum,dims=2)
dLogL_Sum_Block_Epi1 = LogL_Sum_Epi1 - LogL_Unif_Sum_Epi1
dLogL_Sum_Block_Epi1B2 = LogL_Sum_Epi1B2 - LogL_Unif_Sum_Epi1B2

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 8A
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
x = 1:Alg_Num
Colors = ["#6E0D0D","#C785EA","#B159E0","#8723BE"]
Legend = ["MF+S+N","Hyb+S+U", "Hyb+N", "SurNoR"]

# All data
y = dLogL_Sum_Block[1,1,1,:,:] .+ dLogL_Sum_Block[1,1,2,:,:]
dy = std(y,dims=2)/sqrt(Part_Num)
y = mean(y,dims=2)
fig = figure(figsize=(6,6.5)); ax = gca()
for i = 1:Alg_Num
   ax.bar(x[i],y[i],color = Colors[i])
   ax.errorbar(x[i],y[i],yerr=dy[i],color="k",linewidth=1,drawstyle="steps",capsize=3)
end
title("Delta Log Evidence - w.r.t. RC - all epi. for both blocks")
ax.set_xticks(1:Alg_Num)
if n_set == 1
    ax.set_ylim([1400,1900])
elseif n_set == 2
    ax.set_ylim([2000,2300])
elseif n_set == 3
    ax.set_ylim([2300,2700])
end
ax.set_xticklabels(Legend,rotation=90)
ax.set_ylabel("delta Log Evidence");
fig.subplots_adjust(bottom = 0.2, left = 0.15)

# ------------------------------------------------------------------------------
# Save LogL as .MAT for Bayesian Model Selection (KE Stephan, et al. 2009)
# ------------------------------------------------------------------------------
LogL_MAT = mean(LogL_Matrix,dims=5)
LogL_MAT = cat(LogL_MAT[:,:,:,1:Alg_Num,1],
               LogL_Unif_Matrix[:,:,:,1,1], dims=4)

file = matopen(string("data/Log_evidences_simulated/nset_", string(n_set), "/LogL_Matrix.mat"), "w")
write(file, "LogL_Matrix", LogL_MAT)
close(file)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 8B: Expected posterior
# WARNING: Before running this part, in principle, one needs to run the MATLAB
#          code "./data/Log_evidences_simulated/LogL_BMS.m" to use package SPM12 and
#          produce the results and save the file "./data/Log_evidences_simulated/nset_i/BMS_Data.mat"
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
BMS = matread(string("data/Log_evidences_simulated/nset_", string(n_set), "/BMS_Data.mat"))
Colors = ["#6E0D0D","#C785EA","#B159E0","#8723BE", "#474848"]

Legend = ["MF+S+N","Hyb+S+U", "Hyb+N", "SurNoR", "RC"]

fig = figure(figsize=(7,6)); ax = gca()
for i = 1:5
   ax.bar(i,BMS["BMS"]["exp_r"][i], color=Colors[i])
   if (i==SurNoR_ind)
        ax.text(i,BMS["BMS"]["exp_r"][i]+0.01,
                string("ϕ = ",string(round(BMS["BMS"]["pxp"][i],digits=5))),
                fontsize=12, horizontalalignment="center", rotation=0)
   end
end
ax.plot([0,6],[1,1] ./ 5, linestyle="dashed",linewidth=1,color="k")
title("Posterior Probabilities for Different Models - Uniform Prior")
ax.set_xticks(1:5)
ax.set_xticklabels(Legend,rotation=90)
ax.set_yticks(0:0.1:1)
ax.set_yticklabels(0:0.1:1)
ax.set_ylabel("Model Posterior Probability");
ax.set_ylim([0,1])
ax.set_xlim([0.25,5.75])
fig.subplots_adjust(bottom = 0.2, left = 0.15)
