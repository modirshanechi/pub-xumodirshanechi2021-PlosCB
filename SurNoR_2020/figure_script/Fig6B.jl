# The code to generate Fig 6B
using PyPlot
using SurNoR_2020
using Statistics
using HypothesisTests

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# ------------------------------------------------------------------------------
# Training Algorithms
# ------------------------------------------------------------------------------
Agents_SurNoR  = SurNoR_2020.Func_SurNoR_CV_train()
Agents_HybN    = SurNoR_2020.Func_HybN_CV_train()
Agents_Tot_Alg = cat(Agents_SurNoR, Agents_HybN, dims=4)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 6B1 and Fig 6B2: Action preferences in state 3 and 7 in B2E1 and B2E2
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Block = 2
Sub_Num = size(Agents_Tot_Alg)[1]
Alg_Num = size(Agents_Tot_Alg)[4]

Map_S = [1,8,3,5,2,9,10,4,6,7]

n_visit = 1 # Looking at the 1st time visiting each state
for Epi = 1:2
    for State = [3,10]

        # A matrix to save participant's actions at each state
        Matrix_Data = zeros(Sub_Num,1)
        # A matrix to save action probabilities of different algorithms for each participant at each state
        Matrix_alg_P_act = zeros(Sub_Num,4,Alg_Num)
        Trap_set = [2,6,7]

        for Sub = 1:Sub_Num
            # Reading real data
            Agent = Agents_Tot_Alg[Sub,Block,Epi,1]

            Ind = 1:length(Agent.Input.obs)
            Ind = Ind[Agent.Input.obs .== State]

            if Epi >= 2
                Agent_old = Agents_Tot_Alg[Sub,Block,Epi-1,1]
                Ind_old = 1:length(Agent_old.Input.obs)
                Ind_old = Ind_old[Agent_old.Input.obs .== State]
            else
                Ind_old = [1]
            end

            if (length(Ind) < n_visit) || (length(Ind_old) < 1)
                Matrix_Data[Sub,:] .= NaN
                Matrix_alg_P_act[Sub,:,:] .= NaN
            else
                a1 = Agent.Input.act[Ind[n_visit]]  # Taken action
                Matrix_Data[Sub,1] = a1
                for Alg = 1:Alg_Num
                    Agent = Agents_Tot_Alg[Sub,Block,Epi,Alg]
                    Matrix_alg_P_act[Sub,:,Alg] = Agent.P_Action[:,Ind[n_visit]]'
                end
            end
        end

        # Choosing the ones who have been in the particular state of interest
        Ind = 1:Sub_Num
        Ind = Ind[isnan.(Matrix_Data[Ind,1]).== 0]

        Color = ["#474848","#8723BE","#B159E0"]
        fig = figure(figsize=(7,12)); ax = gca()
        x = 1:4
        y = [mean(Matrix_Data[Ind,1] .== 0),
             mean(Matrix_Data[Ind,1] .== 1),
             mean(Matrix_Data[Ind,1] .== 2),
             mean(Matrix_Data[Ind,1] .== 3)]
        ax.bar(x, y, align="center", color = Color[1])
        for Alg = 1:Alg_Num
            x = (1:4) .+ (Alg * 5)
            y = mean(Matrix_alg_P_act[Ind,:,Alg],dims=1)
            ax.bar(x, y[:], align="center", color = Color[Alg+1])
        end
        Legends = ["Data", "SurNoR", "Hyb+N"]
        ax.legend(Legends)

        x = 1:4
        y = [mean(Matrix_Data[Ind,1] .== 0),
             mean(Matrix_Data[Ind,1] .== 1),
             mean(Matrix_Data[Ind,1] .== 2),
             mean(Matrix_Data[Ind,1] .== 3)]
        dy = [std(Matrix_Data[Ind,1] .== 0),
             std(Matrix_Data[Ind,1] .== 1),
             std(Matrix_Data[Ind,1] .== 2),
             std(Matrix_Data[Ind,1] .== 3)] ./ sqrt(length(Ind))
        ax.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")
        for Alg = 1:Alg_Num
            x = (1:4) .+ (Alg * 5)
            y = mean(Matrix_alg_P_act[Ind,:,Alg],dims=1)
            dy = std(Matrix_alg_P_act[Ind,:,Alg],dims=1) ./ sqrt(length(Ind))
            ax.errorbar(x,y[:],yerr=dy[:],color="k",linewidth=1,capsize=5,linestyle ="")
        end
        ax.set_ylim([0,1])
        ax.set_xlim([0.25,14.75])
        ax.set_yticks([0,0.5,1])
        xticks = Array(1:4)
        for Alg = 1:Alg_Num
            append!(xticks,Array((1:4) .+ (Alg * 5)))
        end
        ax.set_xticks(xticks)
        ax.plot([0.25,14.75],[0.25,0.25],"--k")
        ax.set_xticklabels(repeat(["a1","a2","a3","a4"],3))
        ax.set_title(string("Visit number ", string(n_visit), " of state ", string(Map_S[State]), " in B", string(Block), "E", string(Epi), " - N=", string(length(Ind))))

        # Conducting the t-tests
        for Alg = 1:Alg_Num
            y1 = Matrix_alg_P_act[Ind,1,Alg]
            y2 = Matrix_alg_P_act[Ind,4,Alg]
            println("----------------------------------------")
            println("----------------------------------------")
            @show(Legends[Alg+1])
            @show(Map_S[State])
            @show(Block)
            @show(Epi)
            @show OneSampleTTest(y1,y2)
        end
    end
end
