# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for Parameters
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
mutable struct Str_Param
    num_act::Int
    num_obs::Int
    count_decay::Float64            # An additonal parameter for decaying counts for novelty. It is equal to 1 for all analysis in the papaer.
    p_change::Float64               # m in the paper is equal to p_change/(1-p_change)
    ϵ::Float64
    α::Float64                      # ρ_b in the paper
    Δα::Float64                     # δρ in the paper
    β_R::Float64
    β_ratio_N2R::Float64
    γ_RL_R::Float64                 # λ_R in the paper
    γ_RL_N::Float64                 # λ_N in the paper
    λ_elig_R::Float64               # μ_R in the paper
    λ_elig_N::Float64               # μ_R in the paper
    Q_R_0::Float64
    Q_N_0::Float64
    N_prior_sweep::Int              # T_PS in the paper
    w_MF_to_MB_1::Float64           # ω_11 or ω_12 in the paper
    w_MF_to_MB_2::Float64           # ω_0 in the paper
    w_MF_to_MB_Scaler::Float64      # ω_MF in the paper
    γ_RL_R_MB::Float64
    γ_RL_N_MB::Float64
    C_thresh::Int                   # For the case of "trap detector", this is the threshold
    β_ratio_N2R_exploit::Float64    # For testing the effect of novelty for exploitation phases
end
function Str_Param(; num_act=4, num_obs=11, count_decay=1,
                    p_change = 0.9, ϵ = 0.01,
                    β_R=3, β_ratio_N2R=0.5, α = 0.1, Δα = 0,
                    γ_RL_R=0.9, γ_RL_N=0.9, λ_elig_R = 0.9, λ_elig_N = 0.9,
                    Q_R_0 = 0, Q_N_0 = 0, N_prior_sweep = 20,
                    w_MF_to_MB_1=1, w_MF_to_MB_2=1, w_MF_to_MB_Scaler=1,
                    γ_RL_R_MB=-1, γ_RL_N_MB=-1, C_thresh = 50, β_ratio_N2R_exploit = 0)

    Str_Param(num_act, num_obs, count_decay,
               p_change, ϵ, α, Δα,
               β_R, β_ratio_N2R, γ_RL_R, γ_RL_R,
               λ_elig_R, λ_elig_N, Q_R_0, Q_N_0,
               N_prior_sweep, w_MF_to_MB_1, w_MF_to_MB_2, w_MF_to_MB_Scaler,
               γ_RL_R_MB, γ_RL_N_MB, C_thresh, β_ratio_N2R_exploit)
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for hyper parameters: Defining what learning rule is used
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
struct Str_HyperParam
    gamma_form::String                  # Form of adaptation
    surprise_form::String               # Definition of surprise
    novelty_form::String                # Definition of novelty
    exploration_form::String            # Whether uses novelty or MB_Curiosity for exploration
    update_form::String                 # Update rule
    update_form_model_free::String      # Update rule for model free
    MF_surprise_modul::String           # Surprise modulation form of MF
    reward_func::Array{Float64,1}       # Reward indecator
    First_Epi_reward_update::Bool       # Whether update MB Q-values in 1st epi
end
function Str_HyperParam(; gamma_form="satur",
                         surprise_form="BF",
                         novelty_form="log-count",
                         exploration_form="Novelty",
                         update_form="VarSMiLe",
                         update_form_model_free="Q_learning",
                         MF_surprise_modul = "Adapt",
                         reward_func=(R = zeros(11); R[1] = 1; R),
                         First_Epi_reward_update=false)

    Str_HyperParam(gamma_form, surprise_form, novelty_form, exploration_form,
                   update_form, update_form_model_free, MF_surprise_modul,
                   reward_func, First_Epi_reward_update)
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for Inputs
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
mutable struct Str_Input
    obs::Array{Int,1}
    act::Array{Int,1}
end
function Str_Input(; obs=zeros(1), act=zeros(1))

    Str_Input(obs, act)
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for Agent
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
mutable struct Str_Agent
    Sub::Int
    Epi::Int
    Block::Int

    Param::Str_Param
    HyperParam::Str_HyperParam
    Input::Str_Input

    Π::Array{Float64,3}                 # Belief = 3D matrix of αs
    Counter::Array{Float64,2}           # Counts for computing novelty
    Novelty::Array{Float64,2}

    Q_N_MF::Array{Float64,2}            # Model free Q-values for novelty
    Q_R_MF::Array{Float64,2}            # Model free Q-values for reward
    E_N::Array{Float64,2}               # Eligibility trace for novelty
    E_R::Array{Float64,2}               # Eligibility trace for reward

    Q_N_MB::Array{Float64,2}            # Model based Q-values for novelty
    Q_R_MB::Array{Float64,2}            # Model based Q-values for reward
    U_N::Array{Float64,2}               # U-values for prioritized sweeping
    U_R::Array{Float64,2}               # U-values for prioritized sweeping

    Surprise::Array{Float64,1}          # Surprise time-seriese
    Novelty_Seq::Array{Float64,1}       # Novelty time-seriese
    δ_N_Seq::Array{Float64,1}           # NPE time-seriese
    δ_R_Seq::Array{Float64,1}           # RPE time-seriese
    U_N_Seq::Array{Float64,1}           # U_N time-seriese
    U_R_Seq::Array{Float64,1}           # U_R time-seriese
    Gamma::Array{Float64,1}             # γ time-seriese
    t_since_change::Array{Float64,1}    # the expected sequence of time-points since the last change-point

    State::Array{Int,1}                 # "State" time-seriese, defined as combinations of actions and states

    P_Action::Array{Float64,2}          # Output of policy

    LogL::Float64                       # Log likelihood
    LogL_Unif::Float64                  # Log likelihood for random choice model

    Acc_Rate::Float64                   # Accuracy rate

    Est_Act::Array{Int,1}               # The sequence of estimated actions
    Est_Act_P::Array{Float64,1}         # The sequence of probabilities of estimated actions

    Sort_Dist_Model::Array{Float64,2}   # Sorted P_Action
    Sort_Dist_Data::Array{Float64,2}    # Sorted P_Action corresponding to data

    First_Epi::Bool                     # Indicator of 1st episode
end
function Str_Agent(; Sub=1, Epi=1, Block=1,
                    Param=Str_Param(),
                    HyperParam=Str_HyperParam(), Input=Str_Input(),
                    Π=ones(11,44,1), Counter=zeros(11,1), Novelty=zeros(11,1),
                    Q_N_MF=zeros(44,1), Q_R_MF=zeros(44,1),
                    E_N=zeros(44,1), E_R=zeros(44,1),
                    Q_N_MB=zeros(44,1), Q_R_MB=zeros(44,1),
                    U_N=zeros(11,1), U_R=zeros(11,1),
                    Surprise=-ones(1), Novelty_Seq=-ones(1),
                    δ_N_Seq=-ones(1), δ_R_Seq=-ones(1),
                    U_N_Seq=-ones(1), U_R_Seq=-ones(1),Gamma=-ones(1),
                    t_since_change = ones(1),
                    State=zeros(1), P_Action=ones(4,1)/4,
                    LogL=0, LogL_Unif=0, Acc_Rate=0,
                    Est_Act=zeros(1), Est_Act_P=zeros(1),
                    Sort_Dist_Model=ones(4,1)/4, Sort_Dist_Data=ones(4,1)/4,
                    First_Epi=true)

   Str_Agent(Sub, Epi, Block, Param, HyperParam, Input, Π,
             Counter, Novelty, Q_N_MF, Q_R_MF, E_N, E_R,
             Q_N_MB, Q_R_MB, U_N, U_R,
             Surprise, Novelty_Seq, δ_N_Seq, δ_R_Seq, U_N_Seq, U_R_Seq,
             Gamma, t_since_change, State, P_Action, LogL, LogL_Unif, Acc_Rate,
             Est_Act, Est_Act_P, Sort_Dist_Model, Sort_Dist_Data,
             First_Epi)
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure for the graph of the environment
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
struct Str_EnvGraph
    TranMat::Array{Int,2}
    InitObs::Array{Int,2}
    Goal::Int
end
function Str_EnvGraph(; TranMat = zeros(2), InitObs = zeros((1,2)), Goal=0)

    Str_EnvGraph(TranMat, InitObs, Goal)
end


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Structure of the agent, switible for saving as .MAT for MATLAB and EEG analysis
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
mutable struct Str_Agent_MAT
    Sub::Int
    Epi::Int
    Block::Int

    HyperParam::Str_HyperParam
    Input::Str_Input

    Pi::Array{Float64,3}
    Counter::Array{Float64,2}
    Novelty::Array{Float64,2}

    Q_N_MF::Array{Float64,2}
    Q_R_MF::Array{Float64,2}
    E_N::Array{Float64,2}
    E_R::Array{Float64,2}

    Q_N_MB::Array{Float64,2}
    Q_R_MB::Array{Float64,2}
    U_N::Array{Float64,2}
    U_R::Array{Float64,2}

    Surprise::Array{Float64,1}
    Novelty_Seq::Array{Float64,1}
    delta_N_Seq::Array{Float64,1}
    delta_R_Seq::Array{Float64,1}
    U_N_Seq::Array{Float64,1}
    U_R_Seq::Array{Float64,1}
    Gamma::Array{Float64,1}

    State::Array{Int,1}

    P_Action::Array{Float64,2}

    LogL::Float64
    LogL_Unif::Float64

    Acc_Rate::Float64

    Est_Act::Array{Int,1}
    Est_Act_P::Array{Float64,1}

    Sort_Dist_Model::Array{Float64,2}
    Sort_Dist_Data::Array{Float64,2}

    First_Epi::Bool
end
function Str_Agent_MAT(; Agent=Str_Agent)

   Str_Agent_MAT(Agent.Sub, Agent.Epi, Agent.Block, Agent.HyperParam, Agent.Input,
                 Agent.Π, Agent.Counter, Agent.Novelty, Agent.Q_N_MF, Agent.Q_R_MF, Agent.E_N, Agent.E_R,
                 Agent.Q_N_MB, Agent.Q_R_MB, Agent.U_N, Agent.U_R,
                 Agent.Surprise, Agent.Novelty_Seq, Agent.δ_N_Seq, Agent.δ_R_Seq, Agent.U_N_Seq, Agent.U_R_Seq,
                 Agent.Gamma, Agent.State, Agent.P_Action, Agent.LogL, Agent.LogL_Unif, Agent.Acc_Rate,
                 Agent.Est_Act, Agent.Est_Act_P, Agent.Sort_Dist_Model, Agent.Sort_Dist_Data,
                 Agent.First_Epi)
end


mutable struct Str_EEG
    Sub::Int
    Epi::Int
    Block::Int
    obs::Array{Int,1}
    EEG::Array{Float64,2}
end
function Str_EEG(; Sub=1, Epi=1, Block=1,
                   obs=zeros(1), EEG=zeros(1,1) )
    Str_EEG(Sub, Epi, Block,obs, EEG)
end
