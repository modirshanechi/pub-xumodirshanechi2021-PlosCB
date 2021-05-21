# The code to generate Fig 6A
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
Agents_SurNoR = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(3)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 6A1: MF learning rate as a function if Surprise
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
p_c = Agents_SurNoR[1,1,1].Param.p_change   # p_c = m/(m+1) in the paper
m = Agents_SurNoR[1,1,1].Param.p_change / (1 - Agents_SurNoR[1,1,1].Param.p_change)

Surp = 0:0.1:1e6
Gamma = (m .* Surp) ./ (1 .+ m .* Surp)

α = Agents_SurNoR[1,1,1].Param.α        # ρ_b in the paper
Δα = Agents_SurNoR[1,1,1].Param.Δα      # δρ in the paper

α_S = α .+ Δα .* Gamma

y = α_S
x = Surp
fig = figure(figsize=(7,6)); ax = gca()
ax.plot(x,y,color="k")
ax.plot(x,ones(size(x)) .* (α + Δα) ,color="#B71616",linewidth=1,linestyle="--")
ax.plot(x,ones(size(x)) .* (α) ,color="#167CB7",linewidth=1,linestyle="--")
ax.set_xscale("log")
ax.set_xlim(1e0,1e6)
ax.set_ylim(0,0.6)

ax.set_title("MF learning rate vs Surp")
ax.set_xlabel("Surprise")
ax.set_ylabel("Model free learning rate")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 6A3: Surprise histogram
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Sub_Num = 12
Surp = []

for Sub = 1:Sub_Num
    for Epi = 1:5
        for Block = 1:2
            Surp_temp = Agents_SurNoR[Sub,Block,Epi].Surprise
            append!(Surp, Surp_temp[2:end])
        end
    end
end

fig = figure(figsize=(7,6)); ax = gca()
ax.hist(Surp,bins= 10 .^ (-1:0.5:6)  , log=true)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(1e0,1e6)
ax.set_ylim(0,1e4)

ax.set_title("Surprise histogram")
ax.set_xlabel("log Surprise")
ax.set_ylabel("Count")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 6A4: Relative importance of MF to MB
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing Q values for
Mean_diff_ratio_2Parts = zeros((Sub_Num,2,2)) # MF to MB

Epi_Phase_Sets = [[1],[2,3,4,5]]
for Sub = 1:Sub_Num
    for Epi_phase = 1:2
        for Block = 1:2
            DQ_R_MB = []
            DQ_R_MF = []
            DQ_N_MB = []
            DQ_N_MF = []

            for Epi = Epi_Phase_Sets[Epi_phase]
                T = length(Agents_SurNoR[Sub,Block,Epi].Input.obs)

                Q_R_MB = zeros((4,T))
                Q_R_MF = zeros((4,T))

                Q_N_MB = zeros((4,T))
                Q_N_MF = zeros((4,T))

                for t = 1:T
                    Q_R_MB[:,t] = Agents_SurNoR[Sub,Block,Epi].Q_R_MB[4*Agents_SurNoR[Sub,Block,Epi].Input.obs[t] .+ (1:4), t]
                    Q_R_MF[:,t] = Agents_SurNoR[Sub,Block,Epi].Q_R_MF[4*Agents_SurNoR[Sub,Block,Epi].Input.obs[t] .+ (1:4), t]
                    Q_N_MB[:,t] = Agents_SurNoR[Sub,Block,Epi].Q_N_MB[4*Agents_SurNoR[Sub,Block,Epi].Input.obs[t] .+ (1:4), t]
                    Q_N_MF[:,t] = Agents_SurNoR[Sub,Block,Epi].Q_N_MF[4*Agents_SurNoR[Sub,Block,Epi].Input.obs[t] .+ (1:4), t]
                end

                DQ_R_MB_temp = findmax(Q_R_MB,dims=1)[1] .- findmin(Q_R_MB,dims=1)[1]
                DQ_R_MF_temp = findmax(Q_R_MF,dims=1)[1] .- findmin(Q_R_MF,dims=1)[1]

                DQ_N_MB_temp = findmax(Q_N_MB,dims=1)[1] .- findmin(Q_N_MB,dims=1)[1]
                DQ_N_MF_temp = findmax(Q_N_MF,dims=1)[1] .- findmin(Q_N_MF,dims=1)[1]

                append!(DQ_R_MB, DQ_R_MB_temp)
                append!(DQ_R_MF, DQ_R_MF_temp)
                append!(DQ_N_MB, DQ_N_MB_temp)
                append!(DQ_N_MF, DQ_N_MF_temp)
            end

            if Epi_phase != 1
                Mean_diff_ratio_2Parts[Sub,Epi_phase,Block] = mean(DQ_R_MF) / mean(DQ_R_MB)
            else
                if Block == 1
                    β_ratio_N2R = Agents_SurNoR[1,1,1].Param.β_ratio_N2R
                    Mean_diff_ratio_2Parts[Sub,Epi_phase,Block] = (mean(DQ_R_MF) .+ (mean(DQ_N_MF) .* β_ratio_N2R)) /
                                                           (mean(DQ_R_MB) .+ (mean(DQ_N_MB) .* β_ratio_N2R))
                elseif Block == 2
                    β_ratio_N2R = Agents_SurNoR[1,2,1].Param.β_ratio_N2R
                    Mean_diff_ratio_2Parts[Sub,Epi_phase,Block] = (mean(DQ_R_MF) .+ (mean(DQ_N_MF) .* β_ratio_N2R)) /
                                                           (mean(DQ_R_MB) .+ (mean(DQ_N_MB) .* β_ratio_N2R))
                end
            end
        end
    end
end

# Effective weights
w_scale = Agents_SurNoR[1,1,1].Param.w_MF_to_MB_Scaler
w_11 = Agents_SurNoR[1,1,1].Param.w_MF_to_MB_1
w_12 = Agents_SurNoR[1,2,1].Param.w_MF_to_MB_1
w_0 = Agents_SurNoR[1,1,1].Param.w_MF_to_MB_2

Eff_MF_weight_2Parts = deepcopy(Mean_diff_ratio_2Parts)
Eff_MF_weight_2Parts[:,1,1] = Eff_MF_weight_2Parts[:,1,1] .* w_scale .* w_11 ./ (1 - w_11)
Eff_MF_weight_2Parts[:,1,2] = Eff_MF_weight_2Parts[:,1,2] .* w_scale .* w_12 ./ (1 - w_12)
Eff_MF_weight_2Parts[:,2,1:2] = Eff_MF_weight_2Parts[:,2,1:2].* w_scale .* w_0 ./ (1 - w_0)

fig = figure(figsize=(7,6)); ax = gca()

x = 1:4
y = mean(Eff_MF_weight_2Parts,dims=1)[:]
dy = std(Eff_MF_weight_2Parts,dims=1)[:] ./ sqrt(12)
ax.bar(x, y[x])
ax.errorbar(x ,y[x],yerr=dy[x],color="k",linewidth=1,drawstyle="steps",capsize=3,linestyle="")

ax.set_xticks(1:4)
ax.set_xticklabels(["E1","E2to5","E1","E2to5"])
title("Hybrid weights MF to MB")
