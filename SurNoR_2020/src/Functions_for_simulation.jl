# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Sample an Action
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_sample_action(Agent, t::Int; Greedy = false)
    P_Action = Agent.P_Action[:,t]
    if Greedy
        (p, act) = findmax(P_Action)
        act = rand((1:4)[P_Action.==p])
        act = act - 1
        return act
    else
        P_cum = cumsum(P_Action)
        p = rand()
        act = (P_cum[1]<p) + (P_cum[2]<p) + (P_cum[3]<p)
        return act
    end

end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Make Observation
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_make_obs(Agent, t::Int, EnvGraph::Str_EnvGraph)
    obs = Agent.Input.obs[t]
    act = Agent.Input.act[t]
    next_obs = EnvGraph.TranMat[obs+1,act+1]
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Simulating an agent
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_simulate_Agent(EnvGraph;
                             Epi=1, Block=1,
                             Param=Str_Param(),
                             HyperParam=Str_HyperParam(),
                             Agent_Prev=Str_Agent(),
                             Greedy = false,
                             T_stop = 1000)

    Input = Str_Input(; obs = [EnvGraph.InitObs[Epi]], act = [-1])
    Agent = Func_initial_Agent(; Sub=1, Epi=Epi, Block=Block,
                                Param=Param, HyperParam=HyperParam,
                                Input=Input, Agent_Prev=Agent_Prev)

    t = 1
    act = Func_sample_action(Agent,t; Greedy=Greedy)
    Agent.Input.act[t] = act

    State = Func_update_State(Agent, t)
    Agent.State[t] = State

    obs = Func_make_obs(Agent,t,EnvGraph)
    Agent.Input.obs = cat(Agent.Input.obs, obs; dims=1)

    Goal = false
    Stop = false
    while (!Goal)&&(!Stop)
        t += 1

        Stop = (t>=T_stop)
        Goal = (Agent.Input.obs[t] == EnvGraph.Goal)

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
        if (!Agent.First_Epi)||(Goal)||(Block==2)||(HyperParam.First_Epi_reward_update)
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


        if !Goal
            if Agent.First_Epi
                P_Action = Func_compute_P_action(Agent, t; Only_Reward=false)
            else
                P_Action = Func_compute_P_action(Agent, t; Only_Reward=true)
            end
            Agent.P_Action = cat(Agent.P_Action, P_Action; dims=2)

            act = Func_sample_action(Agent,t; Greedy=Greedy)
            Agent.Input.act = cat(Agent.Input.act, act; dims=1)

            obs = Func_make_obs(Agent,t,EnvGraph)
            Agent.Input.obs = cat(Agent.Input.obs, obs; dims=1)
        else
            act = -1
            Agent.Input.act = cat(Agent.Input.act, act; dims=1)
        end

        State = Func_update_State(Agent, t)
        Agent.State = cat(Agent.State, State; dims=1)

        Agent.Novelty_Seq = cat(Agent.Novelty_Seq,
                                  Agent.Novelty[Input.obs[t]+1,t-1];
                                  dims=1)
    end

    Input = Agent.Input
    T = length(Input.obs)

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

    return Agent, Stop
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Simulating blocks
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_simulate_Agent_Blocks(EnvGraph_Array;
                                   Epi_Set=1:5, Block_Set=1:2,
                                   Param=[Str_Param(),Str_Param()],
                                   HyperParam=Str_HyperParam(),
                                   Greedy = false,
                                   Seed = 0,
                                   T_stop = 1000)
    Random.seed!(Seed)

    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents = Array{Str_Agent,2}(undef,N_blo,N_epi)
    for Block = Block_Set
        EnvGraph = EnvGraph_Array[Block]
        for Epi = Epi_Set
            if (Block==1)&&(Epi==1)
                Agent_Epi, Stop = Func_simulate_Agent(EnvGraph;
                                                      Epi=Epi, Block=Block,
                                                      Param=Param[Block],
                                                      HyperParam=HyperParam,
                                                      Greedy = Greedy,
                                                      T_stop = T_stop)
            else
                if Epi==1
                    Agent_Prev = Agents[Block-1,end]
                else
                    Agent_Prev = Agents[Block,Epi-1]
                end

                Agent_Epi, Stop = Func_simulate_Agent(EnvGraph;
                                                      Epi=Epi, Block=Block,
                                                      Param=Param[Block],
                                                      HyperParam=HyperParam,
                                                      Greedy = Greedy,
                                                      Agent_Prev=Agent_Prev,
                                                      T_stop = T_stop)
            end

            if Stop
                return [Agent_Epi]
            end
            Agents[Block,Epi] = Agent_Epi
        end
    end

    return Agents
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# η to params SurNoR
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_param_generator(η0;
                              σ = 0.1*ones(18),
                              η_low = [0,1e-12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              η_up = [0.999,1,Inf,Inf,Inf,0.999,0.999,0.999,0.999,
                                      1,1,Inf,Inf,1,1,1,1,Inf,Inf],
                              Seed = 1,
                              noise_free=true)
    Random.seed!(Seed)
    n_par = length(η0)
    if noise_free
        η = η0
    else
        η = η0 .+ (randn(n_par).*σ)

        for i = 1:n_par
            if η[i] < η_low[i]
                η[i] = η_low[i]
            elseif η[i] > η_up[i]
                η[i] = η_up[i]
            end
        end
    end

    Param1 = Str_Param(; num_act=4, num_obs=11,
                        count_decay=1, Q_R_0 = 0)

    Param1.p_change = η[1]
    Param1.ϵ = η[2]
    Param1.β_R = η[3]
    Param1.β_ratio_N2R = η[4]
    Param1.γ_RL_R = η[6]
    Param1.γ_RL_N = η[7]
    Param1.λ_elig_R = η[8]
    Param1.λ_elig_N = η[9]
    Param1.α = η[10]
    Param1.Δα = η[11]
    Param1.Q_N_0 = η[12]
    Param1.N_prior_sweep = Int64(round(η[13]))
    Param1.w_MF_to_MB_2 = η[14]
    Param1.w_MF_to_MB_1 = η[15]
    Param1.w_MF_to_MB_Scaler = η[17]

    Param2 = deepcopy(Param1)
    Param2.β_ratio_N2R = η[5]
    Param2.w_MF_to_MB_1 = η[16]
    Param2.β_R = η[18]

    Param = [Param1, Param2]
end
