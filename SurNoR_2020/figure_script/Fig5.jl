# The code to generate Fig 5
# The relevant statistics reported in the paper are calculated at the end of the
# file
using PyPlot
using SurNoR_2020
using Statistics
using MAT
using HypothesisTests

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# ------------------------------------------------------------------------------
# Training Algorithms
# ------------------------------------------------------------------------------
for Part_ind = 1:4  # For 4 different repetitions of the optimization
    # Main alternative algorithms for Fig 5
    Agents_MBQ0S    = SurNoR_2020.Func_MBQ0S_CV_train(Part_ind=Part_ind)        # MB + S + OI
    Agents_MBSExp   = SurNoR_2020.Func_MBSExp_CV_train(Part_ind=Part_ind)       # MB + S + U
    Agents_MB_Leaky = SurNoR_2020.Func_MB_Leaky_CV_train(Part_ind=Part_ind)     # MB + N
    Agents_MBS      = SurNoR_2020.Func_MBS_CV_train(Part_ind=Part_ind)          # MB + S + N
    Agents_MFQ0     = SurNoR_2020.Func_MFQ0_CV_train(Part_ind=Part_ind)         # MF + OI
    Agents_MFSExp   = SurNoR_2020.Func_MFSExp_CV_train(Part_ind=Part_ind)       # MF + S + U
    Agents_MFN      = SurNoR_2020.Func_MFN_CV_train(Part_ind=Part_ind)          # MF + N
    Agents_MFNS     = SurNoR_2020.Func_MFNS_CV_train(Part_ind=Part_ind)         # MF + S + N
    Agents_HybQ0S   = SurNoR_2020.Func_HybQ0S_CV_train(Part_ind=Part_ind)       # Hyb + S + OI
    Agents_HybSExp  = SurNoR_2020.Func_HybSExp_CV_train(Part_ind=Part_ind)      # Hyb + S + U
    Agents_HybN     = SurNoR_2020.Func_HybN_CV_train(Part_ind=Part_ind)         # Hyb + N
    Agents_SurNoR   = SurNoR_2020.Func_SurNoR_CV_train(Part_ind=Part_ind)       # SurNoR (Hyb + S + N)

    # Control algorithms
    Agents_HybSTrapDet           = SurNoR_2020.Func_HybSTrapDet_CV_train(Part_ind=Part_ind)         # Binary Novelty 1
    Agents_HybSTrapDet2          = SurNoR_2020.Func_HybSTrapDet2_CV_train(Part_ind=Part_ind)        # Binary Novelty 2
    Agents_SurNoR_NovInExploit   = SurNoR_2020.Func_SurNoR_NovInExploit_CV_train(Part_ind=Part_ind) # SurNoR with β_N ≧ 0 also in Epi > 1

    if Part_ind == 1
        global Agents_Tot_Alg = cat(Agents_MBQ0S, Agents_MBSExp, Agents_MB_Leaky, Agents_MBS,
                                    Agents_MFQ0, Agents_MFSExp, Agents_MFN, Agents_MFNS,
                                    Agents_HybQ0S, Agents_HybSExp, Agents_HybN, Agents_SurNoR,
                                    Agents_HybSTrapDet, Agents_HybSTrapDet2, Agents_SurNoR_NovInExploit,  dims=4)

    else
        Agents_Tot_Alg_temp = cat(Agents_MBQ0S, Agents_MBSExp, Agents_MB_Leaky, Agents_MBS,
                                  Agents_MFQ0, Agents_MFSExp, Agents_MFN, Agents_MFNS,
                                  Agents_HybQ0S, Agents_HybSExp, Agents_HybN, Agents_SurNoR,
                                  Agents_HybSTrapDet, Agents_HybSTrapDet2, Agents_SurNoR_NovInExploit, dims=4)

        Agents_Tot_Alg = cat(Agents_Tot_Alg, Agents_Tot_Alg_temp, dims=5)
    end
end
SurNoR_ind = 12   # index corresponding to SurNoR


# Extracting Log-likelihoods
Sub_Num = 12
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
# Fig 5A: Log Evidences
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Alg_Num_plot = 12
x = 1:Alg_Num_plot
Colors = ["#69C6E5","#45A2C1","#167CB7","#0F5680",
          "#E56969","#C14545","#B71616","#6E0D0D",
          "#D7A8F0","#C785EA","#B159E0","#8723BE"]

Legend = ["MB+S+OI", "MB+S+U", "Leak+N", "MB+S+N",
          "MF+OI" , "MF+S+U", "MF+N", "MF+S+N",
          "Hyb+S+OI", "Hyb+S+U", "Hyb+N", "SurNoR"]

# All data
y = dLogL_Sum_Block[1,1,1,:,:] .+ dLogL_Sum_Block[1,1,2,:,:]
dy = std(y,dims=2)/sqrt(Part_Num)
y = mean(y,dims=2)
fig = figure(figsize=(6,6.5)); ax = gca()
for i = 1:Alg_Num_plot
   ax.bar(x[i],y[i],color = Colors[i])
   ax.errorbar(x[i],y[i],yerr=dy[i],color="k",linewidth=1,drawstyle="steps",capsize=3)
end
title("Delta Log Evidence - w.r.t. RC - all epi. for both blocks")
ax.set_xticks(1:Alg_Num_plot)
ax.set_ylim([450,1300])
ax.set_xticklabels(Legend,rotation=90)
ax.set_ylabel("delta Log Evidence");
fig.subplots_adjust(bottom = 0.2, left = 0.15)

# Episode 1 of block 1
y = dLogL_Sum_Block_Epi1
dy = std(y,dims=3)/sqrt(Part_Num)
y = mean(y,dims=3)
fig = figure(figsize=(6,6.5)); ax = gca()
for i = 1:Alg_Num_plot
   ax.bar(x[i],y[i],color = Colors[i])
   ax.errorbar(x[i],y[i],yerr=dy[i],color="k",linewidth=1,drawstyle="steps",capsize=3)
end
title("Delta Log Evidence - w.r.t. RC - B1E1")
ax.set_xticks(1:Alg_Num_plot)
ax.set_ylim([-20,400])
ax.set_xticklabels(Legend,rotation=90)
title("Delta Log Evidence - w.r.t. RC - B1E1")
fig.subplots_adjust(bottom = 0.2, left = 0.15)

# Episode 1 of block 2
y = dLogL_Sum_Block_Epi1B2
dy = std(y,dims=3)/sqrt(Part_Num)
y = mean(y,dims=3)
fig = figure(figsize=(6,6.5)); ax = gca()
for i = 1:Alg_Num_plot
   ax.bar(x[i],y[i],color = Colors[i])
   ax.errorbar(x[i],y[i],yerr=dy[i],color="k",linewidth=1,drawstyle="steps",capsize=5)
end
title("Delta Log Evidence - w.r.t. RC - B2E1")
ax.set_xticks(1:Alg_Num_plot)
ax.set_ylim([-10,200])
ax.set_xticklabels(Legend,rotation=90)
ax.set_ylabel("delta Log Evidence");
fig.subplots_adjust(bottom = 0.2, left = 0.15)

# ------------------------------------------------------------------------------
# Save LogL as .MAT for Bayesian Model Selection (KE Stephan, et al. 2009)
# ------------------------------------------------------------------------------
LogL_MAT = mean(LogL_Matrix,dims=5)
LogL_MAT = cat(LogL_MAT[:,:,:,1:Alg_Num_plot,1],
               LogL_Unif_Matrix[:,:,:,1,1], dims=4)

file = matopen("./data/Log_evidences/LogL_Matrix.mat", "w")
write(file, "LogL_Matrix", LogL_MAT)
close(file)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 5B: Expected posterior
# WARNING: Before running this part, in principle, one needs to run the MATLAB
#          code "./data/Log_evidences/LogL_BMS.m" to use package SPM12 and
#          produce the results and save the file "./data/Log_evidences/BMS_Data.mat"
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
BMS = matread("./data/Log_evidences/BMS_Data.mat")

Colors = ["#69C6E5","#45A2C1","#167CB7","#0F5680",
          "#E56969","#C14545","#B71616","#6E0D0D",
          "#D7A8F0","#C785EA","#B159E0","#8723BE", "#474848"]

Legend = ["MB+S+OI", "MB+S+U", "Leak+N", "MB+S+N",
          "MF+OI" , "MF+S+U", "MF+N", "MF+S+N",
          "Hyb+S+OI", "Hyb+S+U", "Hyb+N", "SurNoR", "RC"]

fig = figure(figsize=(7,6)); ax = gca()
for i = 1:13
   ax.bar(i,BMS["BMS"]["exp_r"][i], color=Colors[i])
   if (i==SurNoR_ind)
        ax.text(i,BMS["BMS"]["exp_r"][i]+0.01,
                string("ϕ = ",string(round(BMS["BMS"]["pxp"][i],digits=2))),
                fontsize=12, horizontalalignment="center", rotation=0)
   end
end
ax.plot([0,14],[1,1] ./ 13, linestyle="dashed",linewidth=1,color="k")
title("Posterior Probabilities for Different Models")
ax.set_xticks(1:13)
ax.set_xticklabels(Legend,rotation=90)
ax.set_yticks(0:0.1:1)
ax.set_yticklabels(0:0.1:1)
ax.set_ylabel("Model Posterior Probability");
ax.set_ylim([0,0.8])
ax.set_xlim([0.25,13.75])
fig.subplots_adjust(bottom = 0.2, left = 0.15)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 5C: Accuracy rates
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reading the accuracy rates
Sub_Num = 12
Acc_Matrix = zeros((Sub_Num,5,2,Alg_Num,Part_Num))      # To save accuracy rates
NCor_Matrix = zeros((Sub_Num,5,2,Alg_Num,Part_Num))     # To save number of correctly predicted actions
NTot_Matrix = zeros((Sub_Num,5,2,Alg_Num,Part_Num))     # To save total number of actions

for Sub = 1:Sub_Num
    for Epi = 1:5
        for Block = 1:2
            for Alg = 1:Alg_Num
                for Part_ind = 1:Part_Num
                    Acc_Matrix[Sub,Epi,Block,Alg,Part_ind] =
                            Agents_Tot_Alg[Sub,Block,Epi,Alg,Part_ind].Acc_Rate
                    NCor_Matrix[Sub,Epi,Block,Alg,Part_ind] =
                            sum(Agents_Tot_Alg[Sub,Block,Epi,Alg,Part_ind].Sort_Dist_Data[4,:])
                    NTot_Matrix[Sub,Epi,Block,Alg,Part_ind] =
                            length(Agents_Tot_Alg[Sub,Block,Epi,Alg,Part_ind].Est_Act)
                end
            end
        end
    end
end

Acc_Mean_4Parts = mean(Acc_Matrix,dims=1)
Acc_Std_4Parts = std(Acc_Matrix,dims=1)/sqrt(Sub_Num)

Acc_Mean = mean(Acc_Mean_4Parts,dims=5)
Acc_Std = sqrt.(mean(Acc_Std_4Parts,dims=5).^2 .+ var(Acc_Mean_4Parts,dims=5)./4)

# Computing entropy
Average_Entropy_Matrix = zeros((Sub_Num,5,2,Alg_Num,Part_Num))
for Sub = 1:Sub_Num
    for Epi = 1:5
        for Block = 1:2
            for Alg = 1:Alg_Num
                for Part_ind = 1:Part_Num
                    temp = Agents_Tot_Alg[Sub,Block,Epi,Alg,Part_ind].P_Action
                    temp = - temp .* log.(temp)
                    Average_Entropy_Matrix[Sub,Epi,Block,Alg,Part_ind] = mean(sum(temp,dims=1))
                end
            end
        end
    end
end

Average_Entropy_Mean_4Parts = mean(Average_Entropy_Matrix,dims=1)
Average_Entropy_Std_4Parts = std(Average_Entropy_Matrix,dims=1)/sqrt(Sub_Num)

Average_Entropy_Mean = mean(Average_Entropy_Mean_4Parts,dims=5)
Average_Entropy_Std = sqrt.(mean(Average_Entropy_Std_4Parts,dims=5).^2 .+ var(Average_Entropy_Mean_4Parts,dims=5)./4)


Alg_plot = SurNoR_ind  # Choosing SurNoR

fig, (ax1, ax2) = subplots(1, 2, figsize=(10,6.5))

x = 1:0.5:3

y = 100 * Acc_Mean[1,:,1,Alg_plot,1]
dy = 100 * Acc_Std[1,:,1,Alg_plot,1]
ax1.bar(x, y, width=0.3, align="center", color = "#8723BE")
ax1.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")

y = 100 * Acc_Mean[1,:,2,Alg_plot,1]
dy = 100 * Acc_Std[1,:,2,Alg_plot,1]
ax2.bar(x, y, width=0.3, align="center", color = "#8723BE")
ax2.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")

ax1.set_xticks(x)
ax1.set_xticklabels(["1", "2", "3", "4", "5"])
ax2.set_xticks(x)
ax2.set_xticklabels(["1", "2", "3", "4", "5"])
ax1.set_yticks([0,25,50,75,100])
ax1.set_yticklabels(["0%","25%", "50%", "75%", "100%"])
ax2.set_yticks([])
ax2.set_yticklabels([])

ax1.set_ylim(0,100); ax2.set_ylim(0,100);
ax1.set_xlim(0.5,3.5); ax2.set_xlim(0.5,3.5);
ax1.set_xlabel("Episode"); ax2.set_xlabel("Episode")
ax1.set_ylabel("Accuracy Rate of SurNoR in Predicting Actions");
ax1.set_title("1st Block"); ax2.set_title("2nd Block")

ax12 = ax1.twinx()
y = Average_Entropy_Mean[1,:,1,Alg_plot,1]
dy = Average_Entropy_Std[1,:,1,Alg_plot,1]
ax12.errorbar(x, y[:], yerr=dy[:],color="#2D2D2D", linestyle="dashed",capsize=5)

ax22 = ax2.twinx()
y = Average_Entropy_Mean[1,:,2,Alg_plot,1]
dy = Average_Entropy_Std[1,:,2,Alg_plot,1]
ax22.errorbar(x, y[:], yerr=dy[:],color="#2D2D2D", linestyle="dashed",capsize=5)

ax12.set_ylim([0,1.3])
ax22.set_ylim([0,1.3])
ax12.set_yticks([])
ax12.set_yticklabels([])


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reporting the accuracy rates of different phases of SurNoR
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reporting the accuracy after the first time finding the goal
NTot_AG = sum(sum(NTot_Matrix[:,2:end,:,SurNoR_ind:SurNoR_ind,:],dims=2),dims=3) .+
          NTot_Matrix[:,1:1,2:2,SurNoR_ind:SurNoR_ind,:]
NCor_AG = sum(sum(NCor_Matrix[:,2:end,:,SurNoR_ind:SurNoR_ind,:],dims=2),dims=3) .+
          NCor_Matrix[:,1:1,2:2,SurNoR_ind:SurNoR_ind,:]
Acc_AG = NCor_AG ./ NTot_AG

Acc_AG_Mean_4Parts = mean(Acc_AG,dims=1)
Acc_AG_Std_4Parts = std(Acc_AG,dims=1)/sqrt(Sub_Num)

Acc_AG_Mean = mean(Acc_AG_Mean_4Parts,dims=5)
Acc_AG_Std = sqrt.(mean(Acc_AG_Std_4Parts,dims=5).^2 .+ var(Acc_AG_Mean_4Parts,dims=5)./4)

println("Accuracy rate after the 1st time finiding the goal:")
@show Acc_AG_Mean[1];
println("SEM of the accuracy rate after the 1st time finiding the goal:")
@show Acc_AG_Std[1];
println("Total number of actions after the 1st time finiding the goal:")
@show sum(NTot_AG)/Part_Num

# Reporting the accuracy before the first time finding the goal
NTot_BG = sum(sum(NTot_Matrix[:,1:1,1:1,SurNoR_ind:SurNoR_ind,:],dims=2),dims=3)
NCor_BG = sum(sum(NCor_Matrix[:,1:1,1:1,SurNoR_ind:SurNoR_ind,:],dims=2),dims=3)
Acc_BG = NCor_BG ./ NTot_BG

Acc_BG_Mean_4Parts = mean(Acc_BG,dims=1)
Acc_BG_Std_4Parts = std(Acc_BG,dims=1)/sqrt(Sub_Num)

Acc_BG_Mean = mean(Acc_BG_Mean_4Parts,dims=5)
Acc_BG_Std = sqrt.(mean(Acc_BG_Std_4Parts,dims=5).^2 .+ var(Acc_BG_Mean_4Parts,dims=5)./4)


println("Accuracy rate before the 1st time finiding the goal (B1E1):")
@show Acc_BG_Mean[1];
println("SEM of the accuracy rate before the 1st time finiding the goal (B1E1):")
@show Acc_BG_Std[1];
println("Total number of actions before the 1st time finiding the goal (B1E1):")
@show sum(NTot_BG)/Part_Num

# Reporting the accuracy of all actions
NTot_All = sum(sum(NTot_Matrix[:,:,:,SurNoR_ind:SurNoR_ind,:],dims=2),dims=3)
NCor_All = sum(sum(NCor_Matrix[:,:,:,SurNoR_ind:SurNoR_ind,:],dims=2),dims=3)
Acc_All = NCor_All ./ NTot_All

Acc_All_Mean_4Parts = mean(Acc_All,dims=1)
Acc_All_Std_4Parts = std(Acc_All,dims=1)/sqrt(Sub_Num)

Acc_All_Mean = mean(Acc_All_Mean_4Parts,dims=5)
Acc_All_Std = sqrt.(mean(Acc_All_Std_4Parts,dims=5).^2 .+ var(Acc_All_Mean_4Parts,dims=5)./4)

println("Total accuracy rate:")
@show Acc_All_Mean[1];
println("SEM of the total accuracy rate:")
@show Acc_All_Std[1];
println("Total number of actions:")
@show sum(NTot_All)/Part_Num

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reporting the accuracy rates of other exploration strategies
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
OI_Exp_index = [1, 5, 9]
U_Exp_index = [2, 6, 10]

Acc_Matrix_Mean4Parts = mean(Acc_Matrix,dims=5)

println("********************************************")
println("OI:")
println("********************************************")
println("SurNoR vs MB+S+OI:")
@show OneSampleTTest(Acc_Matrix_Mean4Parts[:,1,1,SurNoR_ind,1],Acc_Matrix_Mean4Parts[:,1,1,OI_Exp_index[1],1])
println("--------------------------------------------")
println("SurNoR vs MF+OI:")
@show OneSampleTTest(Acc_Matrix_Mean4Parts[:,1,1,SurNoR_ind,1],Acc_Matrix_Mean4Parts[:,1,1,OI_Exp_index[2],1])
println("--------------------------------------------")
println("SurNoR vs Hyb+S+OI:")
@show OneSampleTTest(Acc_Matrix_Mean4Parts[:,1,1,SurNoR_ind,1],Acc_Matrix_Mean4Parts[:,1,1,OI_Exp_index[3],1])
println("********************************************")
println("OI:")
println("********************************************")
println("SurNoR vs MB+S+U:")
@show OneSampleTTest(Acc_Matrix_Mean4Parts[:,1,1,SurNoR_ind,1],Acc_Matrix_Mean4Parts[:,1,1,U_Exp_index[1],1])
println("--------------------------------------------")
println("SurNoR vs MF+S+U:")
@show OneSampleTTest(Acc_Matrix_Mean4Parts[:,1,1,SurNoR_ind,1],Acc_Matrix_Mean4Parts[:,1,1,U_Exp_index[2],1])
println("--------------------------------------------")
println("SurNoR vs Hyb+S+U:")
@show OneSampleTTest(Acc_Matrix_Mean4Parts[:,1,1,SurNoR_ind,1],Acc_Matrix_Mean4Parts[:,1,1,U_Exp_index[3],1])
println("--------------------------------------------")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Reporting the difference in log-evidence between SurNoR and control algorithms
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
y = dLogL_Sum_Block[1,1,1,:,:] .+ dLogL_Sum_Block[1,1,2,:,:]
dy = std(y,dims=2)/sqrt(Part_Num)
y = mean(y,dims=2)

println("--------------------------------------------")
println("SurNoR vs. Binary Novelty 1:")
println("Mean:")
@show (y[SurNoR_ind] - y[SurNoR_ind+1])
println("SEM:")
@show sqrt(dy[SurNoR_ind]^2 + dy[SurNoR_ind+1]^2)

println("--------------------------------------------")
println("SurNoR vs. Binary Novelty 2:")
println("Mean:")
@show (y[SurNoR_ind] - y[SurNoR_ind+2])
println("SEM:")
@show sqrt(dy[SurNoR_ind]^2 + dy[SurNoR_ind+2]^2)

println("--------------------------------------------")
println("SurNoR vs. SurNoR with β_N ≧ 0 also in Epi > 1:")
println("Mean:")
@show (y[SurNoR_ind] - y[SurNoR_ind+3])
println("SEM:")
@show sqrt(dy[SurNoR_ind]^2 + dy[SurNoR_ind+3]^2)

# Only 1st episode
y = dLogL_Sum_Block_Epi1
dy = std(y,dims=3)/sqrt(Part_Num)
y = mean(y,dims=3)

println("--------------------------------------------")
println("--------------------------------------------")
println("SurNoR vs. Binary Novelty 1 in B1E1:")
println("Mean:")
@show (y[SurNoR_ind] - y[SurNoR_ind+1])
println("SEM:")
@show sqrt(dy[SurNoR_ind]^2 + dy[SurNoR_ind+1]^2)

println("--------------------------------------------")
println("SurNoR vs. Binary Novelty 2 in B1E1:")
println("Mean:")
@show (y[SurNoR_ind] - y[SurNoR_ind+2])
println("SEM:")
@show sqrt(dy[SurNoR_ind]^2 + dy[SurNoR_ind+2]^2)
