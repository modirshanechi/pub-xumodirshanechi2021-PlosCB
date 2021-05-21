# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing D_KL
# The function compute D_KL between 2 Dirichlet dists
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_D_KL(α::Array{Float64,1}, β::Array{Float64,1})
    α_0 = sum(α)
    β_0 = sum(β)
    D1 = + lgamma(α_0) - sum(lgamma.(α))
    D2 = - lgamma(β_0) + sum(lgamma.(β))
    D3 = sum( ( α .- β ) .* ( digamma.(α) .- digamma(α_0) ) )
    D = D1 + D2 + D3
    return D
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing surprise
# The function compute surprise of observing x at time t
# {s(t-1) -> x(t)} -> S(t)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_surprise(Agent, t::Int)
    x = Agent.Input.obs[t]
    s = Agent.State[t-1]
    α = Agent.Π[:, s+1, t-1]
    ϵ = Agent.Param.ϵ

    if Agent.HyperParam.surprise_form == "BF"
        p_x = α[x+1]/(sum(α))
        S = 1/p_x
        return S

    elseif Agent.HyperParam.surprise_form == "Sh"
        p_x = α[x+1]/(sum(α))
        S = - log(p_x)
        return S

    elseif Agent.HyperParam.surprise_form == "CC"
        β = ones(size(α)) .* ϵ
        β[x+1] = ϵ + 1
        S = Func_compute_D_KL(α, β)
        return S
    end

    error("Surprise format is not known!")

end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing gamma
# The function compute gamma of corresponding surprise at time t
# S(t) -> gamma(t)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_gamma(Agent, t::Int)
    p_c = Agent.Param.p_change

    if Agent.HyperParam.gamma_form == "fix"
        gamma=p_c
        return gamma

    elseif Agent.HyperParam.gamma_form == "satur"
        S = Agent.Surprise[t]
        m = p_c / (1 - p_c)
        gamma = m * S / (1 + m*S)
        return gamma
    end

    error("Gamma format is not known!")

end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing t_since_change
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_t_since_change(Agent, t::Int)
    gamma = Agent.Gamma[t]
    t_since_change = Agent.t_since_change[t-1]
    return ( (1-gamma)*t_since_change + 1 )
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Counter
# The function update counter after observing x at time t
# x(t) -> Counter(:,t)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_counter(Agent, t::Int)
    x = Agent.Input.obs[t]
    counter = Agent.Counter[:,t-1]
    counter = Agent.Param.count_decay .* counter
    counter[x+1] += 1
    return counter
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update belief
# Having x(t) and gamma(t), the belief is updated
# x(t) and gamma(t)  -> Π(:,:,t)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_belief(Agent, t::Int)
    x = Agent.Input.obs[t]
    s = Agent.State[t-1]

    Π = Agent.Π[:, :, t-1]
    α = Π[:, s+1]

    γ = Agent.Gamma[t]
    ϵ = Agent.Param.ϵ

    if Agent.HyperParam.update_form == "VarSMiLe"
        α = ( (1 - γ) .* α ) .+ ( γ * ϵ )
        α[x+1] += 1
        Π[:, s+1] = α
        return Π

    elseif Agent.HyperParam.update_form == "Leaky"
        γ = Agent.Param.p_change
        α = ( (1 - γ) .* α )
        α[x+1] += 1
        Π[:, s+1] = α
        return Π
    end

    error("Update format is not known!")
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Novelty
# After updating belief and counter, novely vector is computed
# Counter(:,t) or Π(:,:,t) -> Novelty(:,t)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_novelty(Agent, t::Int)
    if Agent.HyperParam.novelty_form == "log-count"
        P_obs = (Agent.Counter[:,t] .+ 1) ./ (sum( Agent.Counter[:,t] .+ 1) )
        Novelty = - log.(P_obs)
        return Novelty

    elseif Agent.HyperParam.novelty_form == "trap-det"
        Novelty = -1 .* (Agent.Counter[:,t] .> Agent.Param.C_thresh)
        return Novelty

    elseif Agent.HyperParam.novelty_form == "trap-det-2"
        Count = Agent.Counter[:,t]
        Novelty = zeros(size(Count))
        ind_Count = sortperm(Count[:],rev=true)[1:Int(Agent.Param.C_thresh)]
        Novelty[ind_Count[Count[ind_Count].>0]] .= -1
        return Novelty
    end

    error("Novelty format is not known!")
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Compute the estimation of the transition probability
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_P_trans(Agent, t::Int)
    Π = Agent.Π[:,:,t]
    P_Tran = zeros(size(Π))

    for j = 1:(size(Π)[2])
        α0 = sum(Π[:,j])
        P_Tran[:,j] = Π[:,j]./α0;
    end

    return P_Tran
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update "State", defined as the combination of real state and action
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_State(Agent, t::Int)
    State = Agent.Input.obs[t] * Agent.Param.num_act + Agent.Input.act[t]
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Q and U Values for model based
# After updating belief and novelt, new Q and U values are computed either for
# novelty or reward
# Π(:,:,t) and Novelty(:,t) -> Q(:,t) and U(:,t)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_prior_sweep(Agent, t::Int, RorN::String)
    if RorN == "Reward"
        U = Agent.U_R[:,t-1]
        Q = zeros(size(Agent.Q_R_MB[:,t-1]))
        Reward = Agent.HyperParam.reward_func[:]
        Reward = repeat(Reward,1,length(Q))
        if Agent.Param.γ_RL_R_MB == -1
            γ = Agent.Param.γ_RL_R
        else
            γ = Agent.Param.γ_RL_R_MB
        end

    elseif RorN == "Novelty"
        U = Agent.U_N[:,t-1]
        Q = zeros(size(Agent.Q_N_MB[:,t-1]))
        Reward = Agent.Novelty[:,t]
        Reward = repeat(Reward,1,length(Q))
        if Agent.Param.γ_RL_N_MB == -1
            γ = Agent.Param.γ_RL_N
        else
            γ = Agent.Param.γ_RL_N_MB
        end

    elseif RorN == "MB_Curiosity"
        U = Agent.U_N[:,t-1]
        Q = zeros(size(Agent.Q_N_MB[:,t-1]))
        Reward = - log.(Func_compute_P_trans(Agent, t))
        if Agent.Param.γ_RL_N_MB == -1
            γ = Agent.Param.γ_RL_N
        else
            γ = Agent.Param.γ_RL_N_MB
        end

    else
        error("RorN format is not known!")
    end

    P_Tran = Func_compute_P_trans(Agent, t)
    V = zeros(size(U))
    p = zeros(size(U))

    A = Agent.Param.num_act

    # Calculating Q
    for i = 1:length(Q)
        Q[i] = sum( (Reward[:,i] .+ (γ .* U) ) .* P_Tran[:,i] )
    end

    # Calculating V
    for i = 1:length(U)
        s_i = (i-1)*A
        V[i] = findmax( Q[ s_i .+ (1:A) ] )[1]
        p[i] = abs( V[i] - U[i] )
    end

    for i_cycle = 1:Agent.Param.N_prior_sweep
        X_i = findmax(p)[2]
        dV = V[X_i] - U[X_i]
        U[X_i] = V[X_i]

        for i = 1:length(Q)
            Q[i] = Q[i] + ( γ * dV * P_Tran[X_i, i] )
        end

        for i = 1:length(U)
            s_i = (i-1)*A
            V[i] = findmax( Q[ s_i .+ (1:A) ] )[1]
            p[i] = abs( V[i] - U[i] )
        end
    end

    return Q, U
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Compute δ, either RPE or NPE
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_δ(Agent, t, RorN)
    if RorN == "Reward"
        Q = Agent.Q_R_MF[:,t-1]
        Reward = Agent.HyperParam.reward_func[:]
        Reward = repeat(Reward,1,length(Q))
        γ = Agent.Param.γ_RL_R

    elseif RorN == "Novelty"
        Q = Agent.Q_N_MF[:,t-1]
        Reward = Agent.Novelty[:,t]                     # WARNING
        Reward = repeat(Reward,1,length(Q))
        γ = Agent.Param.γ_RL_N

    elseif RorN == "MB_Curiosity"
        Q = Agent.Q_N_MF[:,t-1]
        Reward = - log.(Func_compute_P_trans(Agent, t))
        γ = Agent.Param.γ_RL_N

    else
        error("RorN format is not known!")
    end

    state_old = Agent.State[t-1]                        # (s_{t-1},a_{t-1})
    obs_new = Agent.Input.obs[t]                        #  s_{t}

    A = Agent.Param.num_act
    Q_old = Q[ state_old + 1 ]                          # Q(s_{t-1},a_{t-1})
    V_new = findmax( Q[ (obs_new*A) .+ (1:A) ] )[1]     # max_{a'} Q(s_{t},a')
    R_new = Reward[obs_new + 1, state_old + 1]          # R_{t}

    δ = R_new + γ*V_new - Q_old

    return δ
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Eligibility Trace E
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_ElTrace(Agent, t, RorN)
    if RorN == "Reward"
        E = Agent.E_R[:,t-1]
        λ = Agent.Param.λ_elig_R
        γ = Agent.Param.γ_RL_R

    elseif (RorN == "Novelty") || (RorN == "MB_Curiosity")
        E = Agent.E_N[:,t-1]
        λ = Agent.Param.λ_elig_N
        γ = Agent.Param.γ_RL_N

    else
        error("RorN format is not known!")
    end

    state_old = Agent.State[t-1]                        # (s_{t-1},a_{t-1})
    E = (λ*γ) * E
    E[state_old+1] = 1

    return E
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Update Q values for model-free
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_update_Q_MF(Agent, t, RorN)
    if RorN == "Reward"
        Q = Agent.Q_R_MF[:,t-1]
        δ = Agent.δ_R_Seq[t]
        E = Agent.E_R[:,t]

    elseif (RorN == "Novelty") || (RorN == "MB_Curiosity")
        Q = Agent.Q_N_MF[:,t-1]
        δ = Agent.δ_N_Seq[t]
        E = Agent.E_N[:,t]

    else
        error("RorN format is not known!")
    end

    γ_S = Agent.Gamma[t]
    if Agent.HyperParam.MF_surprise_modul == "Adapt"
        α_0 = Agent.Param.α
        Δα = Agent.Param.Δα
        α = α_0 + Δα*γ_S
    elseif Agent.HyperParam.MF_surprise_modul == "DecayingAdapt"
        α_0 = Agent.Param.α
        Δα = Agent.Param.Δα
        t_since_change = Agent.t_since_change[t]
        α = α_0/(1 + Δα*t_since_change)
    else
        error("MF_surprise_modul format is not known!")
    end

    Q = Q .+ ((α*δ) * E)

    return Q
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing Action Probability
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_compute_P_action(Agent, t::Int; Only_Reward=false)
    if Only_Reward
        Q_MB = Agent.Q_R_MB[:,t] .+ ( Agent.Param.β_ratio_N2R_exploit .* Agent.Q_N_MB[:,t] )
        Q_MF = Agent.Q_R_MF[:,t] .+ ( Agent.Param.β_ratio_N2R_exploit .* Agent.Q_N_MF[:,t] )
    else
        Q_MB = Agent.Q_R_MB[:,t] .+ ( Agent.Param.β_ratio_N2R .* Agent.Q_N_MB[:,t] )
        Q_MF = Agent.Q_R_MF[:,t] .+ ( Agent.Param.β_ratio_N2R .* Agent.Q_N_MF[:,t] )
    end

    if Agent.First_Epi
        Q = (Agent.Param.w_MF_to_MB_1 * Agent.Param.w_MF_to_MB_Scaler) .* Q_MF .+
            (1-Agent.Param.w_MF_to_MB_1) .* Q_MB
    else
        Q = (Agent.Param.w_MF_to_MB_2 * Agent.Param.w_MF_to_MB_Scaler) .* Q_MF .+
            (1-Agent.Param.w_MF_to_MB_2) .* Q_MB
    end

    Q = Q[ (Agent.Input.obs[t]*Agent.Param.num_act) .+ (1:Agent.Param.num_act) ]
    Q = (Q .- findmax(Q)[1]) .* Agent.Param.β_R
    P_Action = exp.(Q)
    P_Action = P_Action ./ sum(P_Action)

    return P_Action
end
