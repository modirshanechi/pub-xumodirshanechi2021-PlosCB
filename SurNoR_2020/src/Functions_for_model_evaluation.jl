# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Functions for training each model using its fitted parameters
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MB_Leaky_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "Leaky"

    Param_Path =
    if Part_ind==-1
        Path = string(Param_Path, "MB_Leaky", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MB_Leaky", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end
    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form = "fix"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                  surprise_form=surprise_form,
                                  novelty_form=novelty_form,
                                  update_form=update_form,
                                  update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                        α = 0, Δα = 0,
                        λ_elig_R = 0, λ_elig_N = 0,
                        count_decay=1, Q_R_0 = 0, Q_N_0 = 0,
                        w_MF_to_MB_1=0, w_MF_to_MB_2=0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5

    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.N_prior_sweep = Param_Data[Fold,2+8]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.β_R = Param_Data[Fold,2+9]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                  Sub_Set = Sub_Set,
                                                  Epi_Set=Epi_Set,
                                                  Block_Set=Block_Set,
                                                  Param=deepcopy(Param),
                                                  HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MBQ0S_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])


    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "MBQ0S", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MBQ0S", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end
    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form = "satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free,
                                   First_Epi_reward_update=true)                           # WARNING WARNING WARNING WARNING

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                        α = 0, Δα = 0,
                        λ_elig_R = 0, λ_elig_N = 0,
                        β_ratio_N2R=0, γ_RL_N=0,
                        count_decay=1, Q_R_0 = 0, Q_N_0 = 0,
                        w_MF_to_MB_1=0, w_MF_to_MB_2=0, w_MF_to_MB_Scaler=1)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Fold_Set = [[9,11,12,13],
               [5,6,7,8],
               [1,2,3,4]]

    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.γ_RL_R = Param_Data[Fold,2+4]
        Param1.N_prior_sweep = Param_Data[Fold,2+5]

        Param2 = deepcopy(Param1)
        Param2.β_R = Param_Data[Fold,2+6]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MBS_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "MBS", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MBS", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end
    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form = "satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                          α = 0, Δα = 0,
                          λ_elig_R = 0, λ_elig_N = 0,
                          count_decay=1, Q_R_0 = 0, Q_N_0 = 0,
                          w_MF_to_MB_1=0, w_MF_to_MB_2=0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.N_prior_sweep = Param_Data[Fold,2+8]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.β_R = Param_Data[Fold,2+9]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MBSExp_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "MBSExp", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MBSExp", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end
    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form = "satur"

    exploration_form = "MB_Curiosity"       #### WARNING ####

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                surprise_form=surprise_form,
                                novelty_form=novelty_form,
                                exploration_form=exploration_form,
                                update_form=update_form,
                                update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                          α = 0, Δα = 0,
                          λ_elig_R = 0, λ_elig_N = 0,
                          count_decay=1, Q_R_0 = 0, Q_N_0 = 0,
                          w_MF_to_MB_1=0, w_MF_to_MB_2=0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.N_prior_sweep = Param_Data[Fold,2+8]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.β_R = Param_Data[Fold,2+9]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MFN_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "Leaky"

    if Part_ind==-1
        Path = string(Param_Path, "MFN", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MFN", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="fix"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                          count_decay=1, Q_R_0=0, Δα=0,
                          p_change=0, ϵ=1, N_prior_sweep=0,
                          w_MF_to_MB_1 = 1, w_MF_to_MB_2 = 1)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.β_R = Param_Data[Fold,2+1]
        Param1.β_ratio_N2R = Param_Data[Fold,2+2]
        Param1.γ_RL_R = Param_Data[Fold,2+4]
        Param1.γ_RL_N = Param_Data[Fold,2+5]
        Param1.λ_elig_R = Param_Data[Fold,2+6]
        Param1.λ_elig_N = Param_Data[Fold,2+7]
        Param1.α = Param_Data[Fold,2+8]
        Param1.Q_N_0 = Param_Data[Fold,2+9]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+3]
        Param2.β_R = Param_Data[Fold,2+10]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MFQ0_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "Leaky"

    if Part_ind==-1
        Path = string(Param_Path, "MFQ0", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MFQ0", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="fix"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                        count_decay=1, Q_N_0=0, Δα=0,
                        β_ratio_N2R=0, γ_RL_N=0, λ_elig_N=0,
                        p_change=0, ϵ=1, N_prior_sweep=0,
                        w_MF_to_MB_1 = 1, w_MF_to_MB_2 = 1)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.β_R = Param_Data[Fold,2+1]
        Param1.γ_RL_R = Param_Data[Fold,2+2]
        Param1.λ_elig_R = Param_Data[Fold,2+3]
        Param1.α = Param_Data[Fold,2+4]
        Param1.Q_R_0 = Param_Data[Fold,2+5]

        Param2 = deepcopy(Param1)
        Param2.β_R = Param_Data[Fold,2+6]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MFNS_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "MFNS", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MFNS", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                        count_decay=1, Q_R_0 = 0,
                        w_MF_to_MB_1=1, w_MF_to_MB_2=1, N_prior_sweep=0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Δα = Param_Data[Fold,2+11]
        Param1.Q_N_0 = Param_Data[Fold,2+12]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.β_R = Param_Data[Fold,2+13]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_MFSExp_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "MFSExp", "_BestParams.csv" )
    else
        Path = string(Param_Path, "MFSExp", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    exploration_form = "MB_Curiosity"       #### WARNING ####

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                surprise_form=surprise_form,
                                novelty_form=novelty_form,
                                exploration_form=exploration_form,
                                update_form=update_form,
                                update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                        count_decay=1, Q_R_0 = 0,
                        w_MF_to_MB_1=1, w_MF_to_MB_2=1, N_prior_sweep=0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Δα = Param_Data[Fold,2+11]
        Param1.Q_N_0 = Param_Data[Fold,2+12]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.β_R = Param_Data[Fold,2+13]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_HybN_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "Leaky"

    if Part_ind==-1
        Path = string(Param_Path, "HybN", "_BestParams.csv" )
    else
        Path = string(Param_Path, "HybN", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="fix"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                               surprise_form=surprise_form,
                               novelty_form=novelty_form,
                               update_form=update_form,
                               update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                                count_decay=1, Q_R_0 = 0, Δα=0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Q_N_0 = Param_Data[Fold,2+11]
        Param1.N_prior_sweep = Param_Data[Fold,2+12]
        Param1.w_MF_to_MB_2 = Param_Data[Fold,2+13]
        Param1.w_MF_to_MB_1 = Param_Data[Fold,2+14]
        Param1.w_MF_to_MB_Scaler = Param_Data[Fold,2+16]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.w_MF_to_MB_1 = Param_Data[Fold,2+15]
        Param2.β_R = Param_Data[Fold,2+17]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_HybQ0S_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "HybQ0S", "_BestParams.csv" )
    else
        Path = string(Param_Path, "HybQ0S", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                               surprise_form=surprise_form,
                               novelty_form=novelty_form,
                               update_form=update_form,
                               update_form_model_free=update_form_model_free,
                               First_Epi_reward_update=true)                           # WARNING WARNING WARNING WARNING

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                        count_decay=1, Q_N_0=0,
                        β_ratio_N2R=0, γ_RL_N=0, λ_elig_N=0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.γ_RL_R = Param_Data[Fold,2+4]
        Param1.λ_elig_R = Param_Data[Fold,2+5]
        Param1.α = Param_Data[Fold,2+6]
        Param1.Δα = Param_Data[Fold,2+7]
        Param1.Q_R_0 = Param_Data[Fold,2+8]
        Param1.N_prior_sweep = Param_Data[Fold,2+9]
        Param1.w_MF_to_MB_2 = Param_Data[Fold,2+10]
        Param1.w_MF_to_MB_1 = Param_Data[Fold,2+11]
        Param1.w_MF_to_MB_Scaler = Param_Data[Fold,2+13]

        Param2 = deepcopy(Param1)
        Param2.w_MF_to_MB_1 = Param_Data[Fold,2+12]
        Param2.β_R = Param_Data[Fold,2+14]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_HybSExp_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "HybSExp", "_BestParams.csv" )
    else
        Path = string(Param_Path, "HybSExp", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    exploration_form = "MB_Curiosity"       #### WARNING ####

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                surprise_form=surprise_form,
                                novelty_form=novelty_form,
                                exploration_form=exploration_form,
                                update_form=update_form,
                                update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                                  count_decay=1, Q_R_0 = 0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Δα = Param_Data[Fold,2+11]
        Param1.Q_N_0 = Param_Data[Fold,2+12]
        Param1.N_prior_sweep = Param_Data[Fold,2+13]
        Param1.w_MF_to_MB_2 = Param_Data[Fold,2+14]
        Param1.w_MF_to_MB_1 = Param_Data[Fold,2+15]
        Param1.w_MF_to_MB_Scaler = Param_Data[Fold,2+17]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.w_MF_to_MB_1 = Param_Data[Fold,2+16]
        Param2.β_R = Param_Data[Fold,2+18]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_HybSTrapDet_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "HybSTrapDet", "_BestParams.csv" )
    else
        Path = string(Param_Path, "HybSTrapDet", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form = "trap-det"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                           surprise_form=surprise_form,
                                           novelty_form=novelty_form,
                                           update_form=update_form,
                                           update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                                  count_decay=1, Q_R_0 = 0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Δα = Param_Data[Fold,2+11]
        Param1.Q_N_0 = Param_Data[Fold,2+12]
        Param1.N_prior_sweep = Param_Data[Fold,2+13]
        Param1.w_MF_to_MB_2 = Param_Data[Fold,2+14]
        Param1.w_MF_to_MB_1 = Param_Data[Fold,2+15]
        Param1.w_MF_to_MB_Scaler = Param_Data[Fold,2+17]
        Param1.C_thresh = Param_Data[Fold,2+19]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.w_MF_to_MB_1 = Param_Data[Fold,2+16]
        Param2.β_R = Param_Data[Fold,2+18]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_HybSTrapDet2_CV_train(;Part_ind=-1,
                                 BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                 Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                                 Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "HybSTrapDet2", "_BestParams.csv" )
    else
        Path = string(Param_Path, "HybSTrapDet2", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form = "trap-det-2"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                           surprise_form=surprise_form,
                                           novelty_form=novelty_form,
                                           update_form=update_form,
                                           update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                                  count_decay=1, Q_R_0 = 0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Δα = Param_Data[Fold,2+11]
        Param1.Q_N_0 = Param_Data[Fold,2+12]
        Param1.N_prior_sweep = Param_Data[Fold,2+13]
        Param1.w_MF_to_MB_2 = Param_Data[Fold,2+14]
        Param1.w_MF_to_MB_1 = Param_Data[Fold,2+15]
        Param1.w_MF_to_MB_Scaler = Param_Data[Fold,2+17]
        Param1.C_thresh = Param_Data[Fold,2+19]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.w_MF_to_MB_1 = Param_Data[Fold,2+16]
        Param2.β_R = Param_Data[Fold,2+18]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_SurNoR_NovInExploit_CV_train(;Part_ind=-1,
                               BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                               Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                               Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]])

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "SurNoR_NovInExploit", "_BestParams.csv" )
    else
        Path = string(Param_Path, "SurNoR_NovInExploit", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                                  count_decay=1, Q_R_0 = 0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Δα = Param_Data[Fold,2+11]
        Param1.Q_N_0 = Param_Data[Fold,2+12]
        Param1.N_prior_sweep = Param_Data[Fold,2+13]
        Param1.w_MF_to_MB_2 = Param_Data[Fold,2+14]
        Param1.w_MF_to_MB_1 = Param_Data[Fold,2+15]
        Param1.w_MF_to_MB_Scaler = Param_Data[Fold,2+17]
        Param1.β_ratio_N2R_exploit = Param_Data[Fold,2+19]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.w_MF_to_MB_1 = Param_Data[Fold,2+16]
        Param2.β_R = Param_Data[Fold,2+18]

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_SurNoR_CV_train(;Part_ind=-1,
                               BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                               Param_Path = "./data/Fitted_parameters/CV_3Fold/",
                               Fold_Set = [[9,11,12,13],[5,6,7,8],[1,2,3,4]],
                               w_MF_to_MB = -1)

    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    if Part_ind==-1
        Path = string(Param_Path, "SurNoR", "_BestParams.csv" )
    else
        Path = string(Param_Path, "SurNoR", "_BestParams","_Part", string(Part_ind), ".csv" )
    end
    Param_Data = CSV.read(Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free)

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                                  count_decay=1, Q_R_0 = 0)

    Param2 = deepcopy(Param1)

    Param = [Param1, Param2]

    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5
    N_sub = 12
    N_blo = length(Block_Set)
    N_epi = length(Epi_Set)
    Agents_Tot = Array{Str_Agent,3}(undef,N_sub,N_blo,N_epi)
    for Fold = 1:3
        Sub_Set = Fold_Set[Fold]

        Param1.p_change = Param_Data[Fold,2+1]
        Param1.ϵ = Param_Data[Fold,2+2]
        Param1.β_R = Param_Data[Fold,2+3]
        Param1.β_ratio_N2R = Param_Data[Fold,2+4]
        Param1.γ_RL_R = Param_Data[Fold,2+6]
        Param1.γ_RL_N = Param_Data[Fold,2+7]
        Param1.λ_elig_R = Param_Data[Fold,2+8]
        Param1.λ_elig_N = Param_Data[Fold,2+9]
        Param1.α = Param_Data[Fold,2+10]
        Param1.Δα = Param_Data[Fold,2+11]
        Param1.Q_N_0 = Param_Data[Fold,2+12]
        Param1.N_prior_sweep = Param_Data[Fold,2+13]
        Param1.w_MF_to_MB_2 = Param_Data[Fold,2+14]
        Param1.w_MF_to_MB_1 = Param_Data[Fold,2+15]
        Param1.w_MF_to_MB_Scaler = Param_Data[Fold,2+17]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_Data[Fold,2+5]
        Param2.w_MF_to_MB_1 = Param_Data[Fold,2+16]
        Param2.β_R = Param_Data[Fold,2+18]

        if w_MF_to_MB != -1
            Param1.w_MF_to_MB_1 = w_MF_to_MB
            Param1.w_MF_to_MB_2 = w_MF_to_MB
            Param2.w_MF_to_MB_1 = w_MF_to_MB
            Param2.w_MF_to_MB_2 = w_MF_to_MB
        end

        Param = [Param1, Param2]

        (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                                   Sub_Set = Sub_Set,
                                                   Epi_Set=Epi_Set,
                                                   Block_Set=Block_Set,
                                                   Param=deepcopy(Param),
                                                   HyperParam=HyperParam)

        Agents_Tot[ ((Fold-1)*4) .+ (1:4) ,:,:] .= Agents
    end

    return deepcopy(Agents_Tot)
end

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
function Func_SurNoR_Overall_train_opt_ind_i(i;BehavData = CSV.read("./data/BehavData_SwitchState.csv"),
                                               Sub_Set = [1,2,3,4,5,6,7,8,9,11,12,13],
                                               Param_Path = "./data/Fitted_parameters/Overall/",
                                               w_MF_to_MB = -1)
    # --------------------------------------------------------------------------
    # Model Based Algorithms
    update_form = "VarSMiLe"

    Param_Path = string(Param_Path, "OverAll_SurNoR_BestParams_diff5.csv" )
    Param_Data = CSV.read(Param_Path)

    if update_form == "SMiLe"
        surprise_form="CC"
    else
        surprise_form="BF"
    end

    novelty_form="log-count"
    update_form_model_free = "Q_learning"
    gamma_form="satur"

    # --------------------------------------------------------------------------
    # Defining Hyper Parameters
    HyperParam = Str_HyperParam(; gamma_form=gamma_form,
                                   surprise_form=surprise_form,
                                   novelty_form=novelty_form,
                                   update_form=update_form,
                                   update_form_model_free=update_form_model_free)



    # --------------------------------------------------------------------------
    # Fitting the models
    Block_Set = 1:2
    Epi_Set = 1:5

    # --------------------------------------------------------------------------
    # Defining Parameters
    Param1 = Str_Param(; num_act=4, num_obs=11,
                         count_decay=1, Q_R_0 = 0)

    Param1.p_change = Param_Data[i,1+1]
    Param1.ϵ = Param_Data[i,1+2]
    Param1.β_R = Param_Data[i,1+3]
    Param1.β_ratio_N2R = Param_Data[i,1+4]
    Param1.γ_RL_R = Param_Data[i,1+6]
    Param1.γ_RL_N = Param_Data[i,1+7]
    Param1.λ_elig_R = Param_Data[i,1+8]
    Param1.λ_elig_N = Param_Data[i,1+9]
    Param1.α = Param_Data[i,1+10]
    Param1.Δα = Param_Data[i,1+11]
    Param1.Q_N_0 = Param_Data[i,1+12]
    Param1.N_prior_sweep = Param_Data[i,1+13]
    Param1.w_MF_to_MB_2 = Param_Data[i,1+14]
    Param1.w_MF_to_MB_1 = Param_Data[i,1+15]
    Param1.w_MF_to_MB_Scaler = Param_Data[i,1+17]

    Param2 = deepcopy(Param1)
    Param2.β_ratio_N2R = Param_Data[i,1+5]
    Param2.w_MF_to_MB_1 = Param_Data[i,1+16]
    Param2.β_R = Param_Data[i,1+18]

    Param = [Param1, Param2]

    if w_MF_to_MB != -1
        Param1.w_MF_to_MB_1 = w_MF_to_MB
        Param1.w_MF_to_MB_2 = w_MF_to_MB
        Param2.w_MF_to_MB_1 = w_MF_to_MB
        Param2.w_MF_to_MB_2 = w_MF_to_MB
    end

    (Agents, LogL) = Func_train_set_of_Agents(BehavData;
                                               Sub_Set = Sub_Set,
                                               Epi_Set=Epi_Set,
                                               Block_Set=Block_Set,
                                               Param=deepcopy(Param),
                                               HyperParam=HyperParam)

    Agents_Tot = deepcopy(Agents)

    return deepcopy(Agents_Tot)
end
