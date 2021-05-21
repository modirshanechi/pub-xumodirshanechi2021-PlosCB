using CSV
using Statistics
using SurNoR_2020
using Distributions
using Distributed
using DataFrames
using Random

# ------------------------------------------------------------------------------
# Reading Data and making input
# ------------------------------------------------------------------------------
BehavData = CSV.read("./data/BehavData_SwitchState.csv")

# ------------------------------------------------------------------------------
# Going through updated rules
# ------------------------------------------------------------------------------
for update_form = ["Leaky"]
        if update_form == "SMiLe"
            surprise_form="CC"
        else
            surprise_form="BF"
        end

        novelty_form="log-count"
        update_form_model_free = "Q_learning"
        gamma_form="fix"

        # ----------------------------------------------------------------------
        # Defining Hyper Parameters
        HyperParam = SurNoR_2020.Str_HyperParam(; gamma_form=gamma_form,
                                                surprise_form=surprise_form,
                                                novelty_form=novelty_form,
                                                update_form=update_form,
                                                update_form_model_free=update_form_model_free)

        # --------------------------------------------------------------------------
        # Defining Parameters
        Param1 = SurNoR_2020.Str_Param(; num_act=4, num_obs=11,
                                        count_decay=1, Q_R_0=0, Δα=0,
                                        p_change=0, ϵ=1, N_prior_sweep=0,
                                        w_MF_to_MB_1 = 1, w_MF_to_MB_2 = 1, w_MF_to_MB_Scaler=1)
        Param2 = deepcopy(Param1)

        Param = [Param1, Param2]

        # ------------------------------------------------------------------------------
        # Subject Information
        Block_Set = 1:2
        Fold_Set = [[1,2,3,4,5,6,7,8],
                    [1,2,3,4,9,11,12,13],
                    [5,6,7,8,9,11,12,13]]
        Epi_Set = 1:5

        # ------------------------------------------------------------------------------
        # Agents
        T_Iter = 40
        for Chain = 1:100
            @distributed for Fold = 1:3
                    Random.seed!((Fold-1)*100 + Chain)

                    Sub_Set = Fold_Set[Fold]
                    LogL_Seq = zeros(T_Iter, 1)
                    Param_Seq = zeros(T_Iter, 10)                                                                            # WARNING: SHOULD CHANGE

                    Param_Seq[1,1] = rand(Rayleigh(5) ,1)[1]           # β_R
                    Param_Seq[1,2] = rand(Rayleigh(0.05) ,1)[1]        # β_RN1
                    Param_Seq[1,3] = rand(Rayleigh(0.05) ,1)[1]        # β_RN2
                    Param_Seq[1,4] = 0.8 + 0.19*rand(1)[1]             # γ_R
                    Param_Seq[1,5] = 0.4 + 0.59*rand(1)[1]             # γ_N
                    Param_Seq[1,6] = 0.8 + 0.19*rand(1)[1]             # λ_R
                    Param_Seq[1,7] = 0.4 + 0.59*rand(1)[1]             # λ_N
                    Param_Seq[1,8] = 0.0 + 0.19*rand(1)[1]             # α
                    Param_Seq[1,9] = 0 + 4*rand(1)[1]                  # Q_N_0
                    Param_Seq[1,10] = rand(Rayleigh(5) ,1)[1]          # β_R_2

                    Param1.β_R = Param_Seq[1,1]
                    Param1.β_ratio_N2R = Param_Seq[1,2]
                    Param1.γ_RL_R = Param_Seq[1,4]
                    Param1.γ_RL_N = Param_Seq[1,5]
                    Param1.λ_elig_R = Param_Seq[1,6]
                    Param1.λ_elig_N = Param_Seq[1,7]
                    Param1.α = Param_Seq[1,8]
                    Param1.Q_N_0 = Param_Seq[1,9]

                    Param2 = deepcopy(Param1)
                    Param2.β_ratio_N2R = Param_Seq[1,3]
                    Param2.β_R = Param_Seq[1,10]

                    Param = [Param1, Param2]


                    (Agents, LogL) = SurNoR_2020.Func_train_set_of_Agents(BehavData;
                                                                           Sub_Set = Sub_Set,
                                                                           Epi_Set=Epi_Set,
                                                                           Block_Set=Block_Set,
                                                                           Param=deepcopy(Param),
                                                                           HyperParam=HyperParam)
                    LogL_Seq[1] = LogL
                    start = time()
                    Flag = true
                    t = 2
                    while (t<=T_Iter)&&Flag
                            if mod(t,1)==0
                                    @show "-----------------------------"
                                    @show [update_form, Chain, Fold, t]
                                    elapsed = time() - start
                                    @show elapsed
                                    start = time()
                                    @show "-----------------------------"
                            end
                            Param_temp = Param_Seq[t-1,:]
                            Param_perm_order = shuffle(1:10)                                                                    # WARNING: SHOULD CHANGE
                            for i = 1:length(Param_perm_order)
                                    num_param = Param_perm_order[i]

                                    if num_param==1
                                            # β_R
                                            @show "beta_R"
                                            x = 0.1:0.2:15
                                    elseif num_param==2
                                            # β_RN1
                                            @show "beta_RN1"
                                            x = 0.00:0.005:0.6
                                    elseif num_param==3
                                            # β_RN2
                                            @show "beta_RN2"
                                            x = 0.00:0.005:0.6
                                    elseif num_param==4
                                            # γ_R
                                            @show "γ_R"
                                            x = 0.85:0.01:0.99
                                    elseif num_param==5
                                            # γ_N
                                            @show "γ_N"
                                            x = 0.20:0.05:0.99
                                    elseif num_param==6
                                            # λ_R
                                            @show "λ_R"
                                            x = 0.85:0.01:1
                                    elseif num_param==7
                                            # λ_N
                                            @show "λ_N"
                                            x = 0.20:0.05:1
                                    elseif num_param==8
                                            # α
                                            @show "α"
                                            x = 0.01:0.02:0.2
                                    elseif num_param==9
                                            # Q_N_0
                                            @show "Q_N_0"
                                            x = 0.0:0.25:8
                                    elseif num_param==10
                                            # β_R_2
                                            @show "beta_R_2"
                                            x = 0.1:0.2:15
                                    end

                                    LogL_range = zeros(size(x))
                                    for x_ind = 1:length(x)
                                            Param_temp[num_param] = x[x_ind]

                                            Param1.β_R = Param_temp[1]
                                            Param1.β_ratio_N2R = Param_temp[2]
                                            Param1.γ_RL_R = Param_temp[4]
                                            Param1.γ_RL_N = Param_temp[5]
                                            Param1.λ_elig_R = Param_temp[6]
                                            Param1.λ_elig_N = Param_temp[7]
                                            Param1.α = Param_temp[8]
                                            Param1.Q_N_0 = Param_temp[9]

                                            Param2 = deepcopy(Param1)
                                            Param2.β_ratio_N2R = Param_temp[3]
                                            Param2.β_R = Param_temp[10]

                                            Param = [Param1, Param2]

                                            (Agents, LogL) = SurNoR_2020.Func_train_set_of_Agents(BehavData;
                                                                                                   Sub_Set = Sub_Set,
                                                                                                   Epi_Set=Epi_Set,
                                                                                                   Block_Set=Block_Set,
                                                                                                   Param=deepcopy(Param),
                                                                                                   HyperParam=HyperParam)
                                            LogL_range[x_ind] = LogL
                                    end
                                    if sum(isnan.(LogL_range)) > 0
                                            @show LogL_range
                                            LogL_range[isnan.(LogL_range)] .= -Inf
                                    end
                                    (LogL, x_ind) = findmax(LogL_range)
                                    Param_temp[num_param] = x[x_ind]
                            end

                            Param_Seq[t,:] = Param_temp
                            LogL_Seq[t] = LogL
                            @show LogL

                            Flag = (!(LogL_Seq[t-1] == LogL))&&(!(LogL == -Inf))

                            if (t==T_Iter)||(mod(t,5)==0)||(!Flag)                                                              # WARNING: SHOULD CHANGE
                                    Path = string("src/Optimization/CV_3Fold/CoAscData_MFN_F", string(Fold), "_C",
                                                    string(Chain), ".csv")
                                    Saved_Data = DataFrame(LogL = LogL_Seq[:],
                                                           beta_R_1 = Param_Seq[:,1],
                                                           beta_Ratio_1 = Param_Seq[:,2],
                                                           beta_Ratio_2 = Param_Seq[:,3],
                                                           gamma_R = Param_Seq[:,4],
                                                           gamma_N = Param_Seq[:,5],
                                                           lambda_R = Param_Seq[:,6],
                                                           lambda_N = Param_Seq[:,7],
                                                           alpha = Param_Seq[:,8],
                                                           Q_N_0 = Param_Seq[:,9],
                                                           beta_R_2 = Param_Seq[:,10])
                                   CSV.write(Path, Saved_Data)
                            end
                            t += 1
                    end
            end
        end
end
