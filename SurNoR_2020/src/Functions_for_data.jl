# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Make input file for a subject, block, and epi
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_data_to_input(Exp_data_frame::DataFrames.DataFrame;
                            Sub=1, Epi=1, Block=2)
    selected_data = @where(Exp_data_frame,
                           :env .== Block+2, :subID .== Sub, :epi .== Epi)

    obs = selected_data[:state] .- 1
    act = selected_data[:action] .- 1

    if obs[end] != 0
        obs = cat(obs, 0; dims=1)
        act = cat(act, -1; dims=1)
    end

    return Str_Input(; obs=obs, act=act)
end


function Func_data_to_input(Exp_data_frame::Array{Str_Input,3};
                            Sub=1, Epi=1, Block=2)
    return Exp_data_frame[Sub,Block,Epi]
end


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Initialize an Agent for a specific input
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_initial_Agent(; Sub=1, Epi=1, Block=1,
                             Param=Str_Param(),
                             HyperParam=Str_HyperParam(),
                             Input=Str_Input(),
                             Agent_Prev=Str_Agent())

    First_Epi = (Epi == 1)

    if First_Epi&&(Block==1)
        # Belief
        Π = ones(Param.num_obs, Param.num_obs*Param.num_act, 1)
        Π *= Param.ϵ

        # Counter
        Counter = zeros(Param.num_obs,1)
        Counter[ Input.obs[1] + 1 ] += 1

        # Novelty
        if HyperParam.novelty_form == "log-count"
            P_obs = (Counter .+ 1) ./ (sum( Counter .+ 1) )
            Novelty = - log.(P_obs)

        elseif HyperParam.novelty_form == "inv-count"
            P_obs = (Counter .+ 1) ./ (sum( Counter .+ 1) )
            Novelty = 1 ./ P_obs

        elseif HyperParam.novelty_form == "trap-det"
            Novelty = -1 .* (Counter .> Param.C_thresh)

        elseif HyperParam.novelty_form == "trap-det-2"
            Count = Counter
            Novelty = zeros(size(Count))
            ind_Count = sortperm(Count[:],rev=true)[1:Int(Param.C_thresh)]
            Novelty[ind_Count[Count[ind_Count] .> 0]] .= -1

        else
            error("Novelty format is not known!")
        end

        Q_R_MF = ones(Param.num_obs*Param.num_act, 1) .* Param.Q_R_0
        Q_N_MF = ones(Param.num_obs*Param.num_act, 1) .* Param.Q_N_0
        E_R = zeros(Param.num_obs*Param.num_act, 1)
        E_N = zeros(Param.num_obs*Param.num_act, 1)

        # QR and UR
        if HyperParam.First_Epi_reward_update
            Ur0 = 1 / (1 - Param.γ_RL_R)
        else
            Ur0 = 0
        end
        Q_R_MB = ones(Param.num_obs*Param.num_act, 1) .* Ur0
        U_R = ones(Param.num_obs, 1) .* Ur0
        # QN and UN
        if HyperParam.exploration_form == "Novelty"
            Un00 = mean(Novelty)
        elseif HyperParam.exploration_form == "MB_Curiosity"
            Un00 = log(Param.num_obs)
        end
        if Param.γ_RL_N_MB == -1
            Un0 = Un00 / (1 - Param.γ_RL_N)
        else
            Un0 = Un00 / (1 - Param.γ_RL_N_MB)
        end
        Q_N_MB = ones(Param.num_obs*Param.num_act, 1) .* Un0
        U_N = ones(Param.num_obs, 1) .* Un0

        State = ones(1)
        State *= Input.obs[1]*Param.num_act + Input.act[1]
        P_Action = ones(Param.num_act,1) ./ Param.num_act

        t_since_change = ones(1)
    else
        # Belief
        Π = Agent_Prev.Π[:,:,end:end]

        # Counter
        Counter = Agent_Prev.Counter[:,end:end] .* Param.count_decay
        Counter[ Input.obs[1] + 1 , 1] += 1

        # Novelty
        if HyperParam.novelty_form == "log-count"
            P_obs = (Counter .+ 1) ./ (sum( Counter .+ 1) )
            Novelty = - log.(P_obs)

        elseif HyperParam.novelty_form == "inv-count"
            P_obs = (Counter .+ 1) ./ (sum( Counter .+ 1) )
            Novelty = 1 ./ P_obs

        elseif HyperParam.novelty_form == "trap-det"
            Novelty = -1 .* (Counter .> Param.C_thresh)

        elseif HyperParam.novelty_form == "trap-det-2"
            Count = Counter
            Novelty = zeros(size(Count))
            ind_Count = sortperm(Count[:],rev=true)[1:Int(Param.C_thresh)]
            Novelty[ind_Count[Count[ind_Count] .> 0]] .= -1

        else
            error("Novelty format is not known!")
        end

        Q_R_MF = Agent_Prev.Q_R_MF[:,end:end]
        Q_N_MF = Agent_Prev.Q_N_MF[:,end:end]
        E_R = zeros(Param.num_obs*Param.num_act, 1)
        E_N = zeros(Param.num_obs*Param.num_act, 1)

        Q_R_MB = Agent_Prev.Q_R_MB[:,end:end]
        Q_N_MB = Agent_Prev.Q_N_MB[:,end:end]
        U_R = Agent_Prev.U_R[:,end:end]
        U_N = Agent_Prev.U_N[:,end:end]

        State = [Input.obs[1]*Param.num_act + Input.act[1]]

        # Model based updating
        if First_Epi
            Q_temp_MB = Q_R_MB .+ ( Param.β_ratio_N2R .* Q_N_MB )
            Q_temp_MF = Q_R_MF .+ ( Param.β_ratio_N2R .* Q_N_MF )

            Q_temp = (Param.w_MF_to_MB_1 * Param.w_MF_to_MB_Scaler) .* Q_temp_MF .+
                     (1-Param.w_MF_to_MB_1) .* Q_temp_MB
        else
            Q_temp_MB = Q_R_MB
            Q_temp_MF = Q_R_MF

            Q_temp = (Param.w_MF_to_MB_2 * Param.w_MF_to_MB_Scaler) .* Q_temp_MF .+
                     (1-Param.w_MF_to_MB_2) .* Q_temp_MB
        end

        Q_temp = Q_temp[ (Input.obs[1]*Param.num_act) .+ (1:Param.num_act), 1:1]
        Q_temp = (Q_temp .- findmin(Q_temp)[1]) .* Param.β_R

        P_Action = exp.(Q_temp)
        P_Action = P_Action ./ sum(P_Action)

        t_since_change = ones(1) .* Agent_Prev.t_since_change[end]
    end

    Str_Agent(; Sub=Sub, Epi=Epi, Block=Block,
             Param=Param, HyperParam=HyperParam, Input=Input,
             Π=Π, Counter=Counter, Novelty=Novelty,
             Q_N_MF=Q_N_MF, Q_R_MF=Q_R_MF, E_N=E_N, E_R=E_R,
             Q_N_MB=Q_N_MB, Q_R_MB=Q_R_MB, U_N=U_N, U_R=U_R,
             State=State, P_Action=P_Action,
             t_since_change = t_since_change,
             First_Epi=First_Epi)
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Training an agent
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_train_Agent(; Sub=1, Epi=1, Block=1,
                           Param=Str_Param(),
                           HyperParam=Str_HyperParam(),
                           Input=Str_Input(),
                           Agent_Prev=Str_Agent())

    Agent = Func_initial_Agent(;Sub=Sub, Epi=Epi, Block=Block,
                                Param=Param, HyperParam=HyperParam,
                                Input=Input, Agent_Prev=Agent_Prev)
    T = length(Input.obs)
    for t = 2:T
        S = Func_compute_surprise(Agent, t)
        Agent.Surprise = cat(Agent.Surprise, S; dims=1)

        gamma = Func_compute_gamma(Agent,t)
        Agent.Gamma = cat(Agent.Gamma, gamma; dims=1)

        t_since_change = Func_compute_t_since_change(Agent,t)
        Agent.t_since_change = cat(Agent.t_since_change, t_since_change; dims=1)

        counter = Func_update_counter(Agent, t)
        Agent.Counter = cat(Agent.Counter, counter; dims=2)

        Π = Func_update_belief(Agent, t)
        Agent.Π = cat(Agent.Π, Π; dims=3)

        Novelty = Func_update_novelty(Agent, t)
        Agent.Novelty = cat(Agent.Novelty, Novelty; dims=2)

        # Model free updating
        δ_R = Func_compute_δ(Agent, t, "Reward")
        Agent.δ_R_Seq = cat(Agent.δ_R_Seq, δ_R; dims=1)
        δ_N = Func_compute_δ(Agent, t, HyperParam.exploration_form)
        Agent.δ_N_Seq = cat(Agent.δ_N_Seq, δ_N; dims=1)

        E_R = Func_compute_ElTrace(Agent, t, "Reward")
        Agent.E_R = cat(Agent.E_R, E_R; dims=2)
        E_N = Func_compute_ElTrace(Agent, t, HyperParam.exploration_form)
        Agent.E_N = cat(Agent.E_N, E_N; dims=2)

        Q_R_MF = Func_update_Q_MF(Agent, t, "Reward")
        Agent.Q_R_MF = cat(Agent.Q_R_MF, Q_R_MF; dims=2)
        Q_N_MF = Func_update_Q_MF(Agent, t, HyperParam.exploration_form)
        Agent.Q_N_MF = cat(Agent.Q_N_MF, Q_N_MF; dims=2)

        # Model based updating
        if (!Agent.First_Epi)||(t==T)||(Block==2)||(HyperParam.First_Epi_reward_update)
            Q_R_MB, U_R = Func_update_prior_sweep(Agent, t, "Reward")
        else
            Q_R_MB = Agent.Q_R_MB[:,1]
            U_R = Agent.U_R[:,1]
        end
        Agent.Q_R_MB = cat(Agent.Q_R_MB, Q_R_MB; dims=2)
        Agent.U_R = cat(Agent.U_R, U_R; dims=2)

        Q_N_MB, U_N = Func_update_prior_sweep(Agent, t, HyperParam.exploration_form)
        Agent.Q_N_MB = cat(Agent.Q_N_MB, Q_N_MB; dims=2)
        Agent.U_N = cat(Agent.U_N, U_N; dims=2)

        State = Func_update_State(Agent, t)
        Agent.State = cat(Agent.State, State; dims=1)

        if t<T
            if Agent.First_Epi
                P_Action = Func_compute_P_action(Agent, t; Only_Reward=false)
            else
                P_Action = Func_compute_P_action(Agent, t; Only_Reward=true)
            end

            Agent.P_Action = cat(Agent.P_Action, P_Action; dims=2)
        end

        Agent.Novelty_Seq = cat(Agent.Novelty_Seq,
                                Agent.Novelty[Input.obs[t]+1,t-1];
                                dims=1)
    end

    # --------------------------------------------------------------------------
    # Action Analysis - LogL
    for t = 1:(T-1)
        Agent.LogL += log(Agent.P_Action[Input.act[t]+1 ,t])

        (p, a) = findmax(Agent.P_Action[: ,t])
        if t==1
            Agent.Est_Act[1] = a-1
            Agent.Est_Act_P[1] = p
        else
            Agent.Est_Act = cat(Agent.Est_Act, a-1; dims=1)
            Agent.Est_Act_P = cat(Agent.Est_Act_P, p; dims=1)
        end
    end

    Agent.LogL_Unif = (T-1) * log( 1/Param.num_act )
    # --------------------------------------------------------------------------
    # Action Analysis - Acc Rate
    Agent.Sort_Dist_Data = zeros(size(Agent.P_Action))
    Agent.Sort_Dist_Model = zeros(size(Agent.P_Action))

    for t = 1:(T-1)
        temp = sortperm(Agent.P_Action[:,t])
        Agent.Sort_Dist_Model[:,t] = Agent.P_Action[temp,t]

        Agent.Sort_Dist_Data[Input.act[t] + 1 ,t] = 1
        Agent.Sort_Dist_Data[:,t] = Agent.Sort_Dist_Data[temp,t]

        if (Agent.Sort_Dist_Model[findmax(Agent.Sort_Dist_Data[:,t])[2],t] -
            Agent.Sort_Dist_Model[end,t]) == 0

            Agent.Sort_Dist_Data[:,t] = zeros(size(Agent.Sort_Dist_Data[:,t]))

            Agent.Sort_Dist_Data[Agent.Sort_Dist_Model[:,t] .==
                              Agent.Sort_Dist_Model[end,t],t] .= 1

            Agent.Sort_Dist_Data[:,t] = Agent.Sort_Dist_Data[:,t] ./
                                     sum(Agent.Sort_Dist_Data[:,t])
        end
    end

    Agent.LogL_Unif = (T-1) * log( 1/Param.num_act )
    Agent.Acc_Rate = mean(Agent.Sort_Dist_Data[4,:])

    return Agent
end


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Training a set of agents
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_train_set_of_Agents(BehavData;
                                  Sub_Set = [1], Epi_Set=[1], Block_Set=1:2,
                                  Param=[Str_Param(),Str_Param()],
                                  HyperParam=Str_HyperParam())

    LogL = 0
    N_sub = length(Sub_Set)
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Sub_Ind = 1:length(Sub_Set)
        Sub = Sub_Set[Sub_Ind]
        for Block = Block_Set
            for Epi = Epi_Set
                Input = Func_data_to_input(BehavData; Sub = Sub,
                                           Epi = Epi, Block = Block)
                if (Block==1)&&(Epi==1)
                    Agent_Epi = Func_train_Agent(; Sub=Sub, Epi=Epi,
                                                   Block=Block,
                                                   Param=Param[Block],
                                                   HyperParam=HyperParam,
                                                   Input=Input)
                else
                    if Epi==1
                        Agent_Prev = Agents[Sub_Ind,Block-1,end]
                    else
                        Agent_Prev = Agents[Sub_Ind,Block,Epi-1]
                    end

                    Agent_Epi = Func_train_Agent(; Sub=Sub, Epi=Epi,
                                                   Block=Block,
                                                   Param=Param[Block],
                                                   HyperParam=HyperParam,
                                                   Input=Input,
                                                   Agent_Prev=Agent_Prev)
                end

                Agents[Sub_Ind,Block,Epi] = Agent_Epi
                LogL += Agent_Epi.LogL
            end
        end
    end
    return Agents, LogL
end
