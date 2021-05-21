# The code to generate Fig A in S6 Text
using PyPlot
using SurNoR_2020
using Statistics
PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# ------------------------------------------------------------------------------
# Training Algorithms
# ------------------------------------------------------------------------------
CV = false
if CV
    Agents_SurNoR      = SurNoR_2020.Func_SurNoR_CV_train()
    Agents_SurNoR_MB   = SurNoR_2020.Func_SurNoR_CV_train(w_MF_to_MB = 0)
    Agents_SurNoR_MF   = SurNoR_2020.Func_SurNoR_CV_train(w_MF_to_MB = 1)
    CV_title = "CV"
else
    Agents_SurNoR      = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(3)
    Agents_SurNoR_MB   = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(3; w_MF_to_MB = 0)
    Agents_SurNoR_MF   = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(3; w_MF_to_MB = 1)
    CV_title = ""
end

Agents_Tot_Alg = cat(Agents_SurNoR, Agents_SurNoR_MB, Agents_SurNoR_MF, dims=4)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Panel A: The 2nd time in a state after a bad decision
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Epi = 1
Block = 1
Sub_Num = size(Agents_Tot_Alg)[1]
Alg_Num = size(Agents_Tot_Alg)[4]

Map_S = [1,8,3,5,2,9,10,4,6,7]
n_visit = 1

for State = [1,5,3]
    # a matrix to save [Current action, next state, next-time action, progress, repeat] at each state for each participant:
    Matrix_ASA = zeros(Sub_Num,5)
    # a matrix to save the action probability for different algorithms
    Matrix_alg_P_act = zeros(Sub_Num,4,Alg_Num)
    # a matrix to save the repeating and changing for different algorithms
    Matrix_alg_P_repeat = zeros(Sub_Num,2,Alg_Num)
    Trap_set = [2,6,7]

    for Sub = 1:Sub_Num
        Agent = Agents_Tot_Alg[Sub,Block,Epi,1]

        Ind = 1:length(Agent.Input.obs)
        Ind = Ind[Agent.Input.obs .== State]

        if length(Ind) <= 1
            Matrix_ASA[Sub,:] .= NaN
            Matrix_alg_P_act[Sub,:,:] .= NaN
            Matrix_alg_P_repeat[Sub,:,:] .= NaN
        else
            a1 = Agent.Input.act[Ind[n_visit]]      # the action at the 1st visit
            s1 = Agent.Input.obs[Ind[n_visit]+1]    # the next state
            a2 = Agent.Input.act[Ind[n_visit+1]]    # the action at the 2nd visit

            Matrix_ASA[Sub,1] = a1
            Matrix_ASA[Sub,2] = s1
            Matrix_ASA[Sub,3] = a2

            if s1==State
                Matrix_ASA[Sub,4] = 0       # a1 = neutral action
            elseif s1 ∈ Trap_set
                Matrix_ASA[Sub,4] = -1      # a1 = bad action
            else
                Matrix_ASA[Sub,4] = 1       # a1 = good action
            end

            Matrix_ASA[Sub,5] = 1 * (a1 == a2)  # a1 = a2

            for Alg = 1:Alg_Num
                Agent = Agents_Tot_Alg[Sub,Block,Epi,Alg]
                Matrix_alg_P_act[Sub,:,Alg] = Agent.P_Action[:,Ind[n_visit+1]]'
                Matrix_alg_P_repeat[Sub,1,Alg] = Agent.P_Action[a1+1,Ind[n_visit+1]]
                Matrix_alg_P_repeat[Sub,2,Alg] = 1 - Agent.P_Action[a1+1,Ind[n_visit+1]]
            end
        end
    end

    # Choosing the one with a1=bad action
    Progress = -1
    Ind = 1:Sub_Num
    Ind = Ind[Matrix_ASA[Ind,4].==Progress]

    # Plot
    Color = ["#474848","#8723BE","#0F5680","#6E0D0D"]
    fig = figure(figsize=(14,12)); ax = gca()
    x = 1:2
    y = [mean(Matrix_ASA[Ind,5]),mean(1 .- Matrix_ASA[Ind,5])]
    dy = [std(Matrix_ASA[Ind,5]),std(1 .- Matrix_ASA[Ind,5])] / sqrt(length(Ind))
    ax.bar(x, y, align="center", color = Color[1])
    for Alg = 1:Alg_Num
        x = (1:2) .+ (Alg * 2)
        y = [mean(Matrix_alg_P_repeat[Ind,1,Alg]),mean(1 .- Matrix_alg_P_repeat[Ind,1,Alg])]
        dy = [std(Matrix_alg_P_repeat[Ind,1,Alg]),std(1 .- Matrix_alg_P_repeat[Ind,1,Alg])] / sqrt(length(Ind))
        ax.bar(x, y, align="center", color = Color[Alg+1])
    end
    ax.legend(["Data", "SurNoR", "MB-SurNoR","MF-SurNoR"])

    x = 1:2
    y = [mean(Matrix_ASA[Ind,5]),mean(1 .- Matrix_ASA[Ind,5])]
    dy = [std(Matrix_ASA[Ind,5]),std(1 .- Matrix_ASA[Ind,5])] / sqrt(length(Ind))
    ax.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")
    for Alg = 1:Alg_Num
        x = (1:2) .+ (Alg * 2)
        y = [mean(Matrix_alg_P_repeat[Ind,1,Alg]),mean(1 .- Matrix_alg_P_repeat[Ind,1,Alg])]
        dy = [std(Matrix_alg_P_repeat[Ind,1,Alg]),std(1 .- Matrix_alg_P_repeat[Ind,1,Alg])] / sqrt(length(Ind))
        ax.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")
    end
    ax.set_ylim([0,1])
    ax.set_xlim([0.25,8.75])
    ax.set_xticks(1:8)
    ax.plot([0,9],[0.25,0.25],"--k")
    ax.set_xticklabels(repeat(["Stay","Change"],4))
    ax.set_title(string("After 1st failure in state ", string(Map_S[State]), " - N=", string(length(Ind))))
end


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Panel A: The after n-th good action in state 1
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Epi = 1
Block = 1
Sub_Num = size(Agents_Tot_Alg)[1]
Alg_Num = size(Agents_Tot_Alg)[4]

Map_S = [1,8,3,5,2,9,10,4,6,7]

State = 1

# a matrix to save [Current action, next state, next-time action, progress, repeat] at each state for each participant:
Matrix_ASA = zeros(Sub_Num,5)
# a matrix to save the action probability for different algorithms
Matrix_alg_P_act = zeros(Sub_Num,4,Alg_Num)
# a matrix to save the repeating and changing for different algorithms
Matrix_alg_P_repeat = zeros(Sub_Num,2,Alg_Num)

Trap_set = [2,6,7]
Prog_set = [5,1,8,9,3,1,1,4,10,0]

for n_success = [2,6,10]
    for Sub = 1:Sub_Num
        Agent = Agents_Tot_Alg[Sub,Block,Epi,1]

        Ind = 1:length(Agent.Input.obs)
        Ind = Ind[Agent.Input.obs .== State]
        Ind_succ = Ind[Agent.Input.obs[Ind .+ 1] .== Prog_set[State]]

        if length(Ind_succ) <= n_success
            Matrix_ASA[Sub,:] .= NaN
            Matrix_alg_P_act[Sub,:,:] .= NaN
            Matrix_alg_P_repeat[Sub,:,:] .= NaN
        else
            Ind = Ind[Ind .>= Ind_succ[n_success]]

            a1 = Agent.Input.act[Ind[1]]      # the action at the 1st visit
            s1 = Agent.Input.obs[Ind[1]+1]    # the next state
            a2 = Agent.Input.act[Ind[1+1]]    # the action at the 2nd visit

            Matrix_ASA[Sub,1] = a1
            Matrix_ASA[Sub,2] = s1
            Matrix_ASA[Sub,3] = a2

            if s1==State
                Matrix_ASA[Sub,4] = 0       # a1 = neutral action
            elseif s1 ∈ Trap_set
                Matrix_ASA[Sub,4] = -1      # a1 = bad action
            else
                Matrix_ASA[Sub,4] = 1       # a1 = good action
            end

            Matrix_ASA[Sub,5] = 1 * (a1 == a2)  # a1 = a2

            for Alg = 1:Alg_Num
                Agent = Agents_Tot_Alg[Sub,Block,Epi,Alg]
                Matrix_alg_P_act[Sub,:,Alg] = Agent.P_Action[:,Ind[1+1]]'
                Matrix_alg_P_repeat[Sub,1,Alg] = Agent.P_Action[a1+1,Ind[1+1]]
                Matrix_alg_P_repeat[Sub,2,Alg] = 1 - Agent.P_Action[a1+1,Ind[1+1]]
            end
        end
    end

    # The one with a1 = good
    Progress = 1
    Ind = 1:Sub_Num
    Ind = Ind[Matrix_ASA[Ind,4].==Progress]

    # Plot
    Color = ["#474848","#8723BE","#0F5680","#6E0D0D"]
    fig = figure(figsize=(14,12)); ax = gca()
    x = 1:2
    y = [mean(Matrix_ASA[Ind,5]),mean(1 .- Matrix_ASA[Ind,5])]
    dy = [std(Matrix_ASA[Ind,5]),std(1 .- Matrix_ASA[Ind,5])] / sqrt(length(Ind))
    ax.bar(x, y, align="center", color = Color[1])
    for Alg = 1:Alg_Num
        x = (1:2) .+ (Alg * 2)
        y = [mean(Matrix_alg_P_repeat[Ind,1,Alg]),mean(1 .- Matrix_alg_P_repeat[Ind,1,Alg])]
        dy = [std(Matrix_alg_P_repeat[Ind,1,Alg]),std(1 .- Matrix_alg_P_repeat[Ind,1,Alg])] / sqrt(length(Ind))
        ax.bar(x, y, align="center", color = Color[Alg+1])
    end
    ax.legend(["Data", "SurNoR", "MB-SurNoR","MF-SurNoR"])

    x = 1:2
    y = [mean(Matrix_ASA[Ind,5]),mean(1 .- Matrix_ASA[Ind,5])]
    dy = [std(Matrix_ASA[Ind,5]),std(1 .- Matrix_ASA[Ind,5])] / sqrt(length(Ind))
    ax.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")
    for Alg = 1:Alg_Num
        x = (1:2) .+ (Alg * 2)
        y = [mean(Matrix_alg_P_repeat[Ind,1,Alg]),mean(1 .- Matrix_alg_P_repeat[Ind,1,Alg])]
        dy = [std(Matrix_alg_P_repeat[Ind,1,Alg]),std(1 .- Matrix_alg_P_repeat[Ind,1,Alg])] / sqrt(length(Ind))
        ax.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")
    end
    ax.set_ylim([0,1])
    ax.set_xlim([0.25,8.75])
    ax.set_xticks(1:8)
    ax.plot([0,9],[0.25,0.25],"--k")
    ax.set_xticklabels(repeat(["Stay","Change"],4))
    ax.set_title(string("After success number ", string(n_success), "1st failure in state ", string(Map_S[State]), " - N=", string(length(Ind))))

end
