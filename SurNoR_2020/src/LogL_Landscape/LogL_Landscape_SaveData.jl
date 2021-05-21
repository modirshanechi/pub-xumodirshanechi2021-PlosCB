# Saving the LogL landscape data for S4 Fig
using CSV
using SurNoR_2020
using DataFrames
using Query

# A function to read the parameters' name and their optimization range given
# their index
function Func_if_VarName(num_param)
    if num_param==1
        # P_change              # The name in the codes
        VarName = "m/(1+m)"     # The name in the paper
        x = 0.00:0.0025:0.3
    elseif num_param==2
        # Eps
        VarName = "eps"
        x = -12:0.2:0.2
        x = 10 .^ x
    elseif num_param==3
        # β_R
        VarName = "beta_1"
        x = 0.1:0.2:15
    elseif num_param==4
        # β_RN1
        VarName = "beta_N1"
        x = 0.00:0.005:0.6
    elseif num_param==5
        # β_RN2
        VarName = "beta_N2"
        x = 0.00:0.005:0.6
    elseif num_param==6
        # γ_R
        VarName = "λ_R"
        x = 0.85:0.01:0.99
    elseif num_param==7
        # γ_N
        VarName = "λ_N"
        x = 0.20:0.05:0.99
    elseif num_param==8
        # λ_R
        VarName = "μ_R"
        x = 0.85:0.01:1
    elseif num_param==9
        # λ_N
        VarName = "μ_N"
        x = 0.20:0.05:1
    elseif num_param==10
        # α
        VarName = "ρ_b"
        x = 0.0:0.02:0.2
    elseif num_param==11
        # Δα
        VarName = "delta ρ"
        x = 0.0:0.05:(1-Param1.α)
    elseif num_param==12
        # Q_N_0
        VarName = "Q_N0"
        x = 0.0:0.25:12
    elseif num_param==13
        # N_PS
        VarName = "T_PS"
        x = cat(0:14,15:10:60,dims=1)
    elseif num_param==14
        # w_2
        VarName = "w_0"
        x = 0:0.05:1
    elseif num_param==15
        # w_11
        VarName = "w_11"
        x = 0:0.05:1
    elseif num_param==16
        # w_12
        VarName = "w_12"
        x = 0:0.05:1
    elseif num_param==17
        # w_scaler
        VarName = "w_scale"
        x = 0:0.1:6
    elseif num_param==18
        # β_R_2
        VarName = "beta_2"
        x = 0.1:0.2:15
    end
    return (VarName,x)
end

# ------------------------------------------------------------------------------
# Reading Data and making input
BehavData = CSV.read("./data/BehavData_SwitchState.csv")

# --------------------------------------------------------------------------
# Defining Hyper Parameters
update_form = "VarSMiLe"

if update_form == "SMiLe"
        surprise_form="CC"
else
        surprise_form="BF"
end

novelty_form="log-count"
update_form_model_free = "Q_learning"

# ----------------------------------------------------------------------
# Defining Hyper Parameters
HyperParam = SurNoR_2020.Str_HyperParam(; gamma_form="satur",
                                surprise_form=surprise_form,
                                novelty_form=novelty_form,
                                update_form=update_form,
                                update_form_model_free=update_form_model_free)

# ----------------------------------------------------------------------
# Subject Information
Block_Set = 1:2
Sub_Set = [1,2,3,4,5,6,7,8,9,11,12,13]
Epi_Set = 1:5

# ------------------------------------------------------------------------------
# Loading Parameters
Path_Params = "data/Fitted_parameters/Overall/OverAll_SurNoR_BestParams_diff5.csv"
Param_Data = CSV.read(Path_Params)
Row = 3

# ------------------------------------------------------------------------------
# Agents
Param1 = SurNoR_2020.Str_Param(; num_act=4, num_obs=11,
                                 count_decay=1, Q_R_0 = 0)

Param1.p_change = Param_Data[Row,1+1]
Param1.ϵ = Param_Data[Row,1+2]
Param1.β_R = Param_Data[Row,1+3]
Param1.β_ratio_N2R = Param_Data[Row,1+4]
Param1.γ_RL_R = Param_Data[Row,1+6]
Param1.γ_RL_N = Param_Data[Row,1+7]
Param1.λ_elig_R = Param_Data[Row,1+8]
Param1.λ_elig_N = Param_Data[Row,1+9]
Param1.α = Param_Data[Row,1+10]
Param1.Δα = Param_Data[Row,1+11]
Param1.Q_N_0 = Param_Data[Row,1+12]
Param1.N_prior_sweep = Param_Data[Row,1+13]
Param1.w_MF_to_MB_2 = Param_Data[Row,1+14]
Param1.w_MF_to_MB_1 = Param_Data[Row,1+15]
Param1.w_MF_to_MB_Scaler = Param_Data[Row,1+17]

Param2 = deepcopy(Param1)
Param2.β_ratio_N2R = Param_Data[Row,1+5]
Param2.w_MF_to_MB_1 = Param_Data[Row,1+16]
Param2.β_R = Param_Data[Row,1+18]

Param = [Param1, Param2]

Param_perm_order = 1:18
for i = 11
    num_param_i = Param_perm_order[i]

    Param_temp = zeros(18,)
    for i_dummy = 1:18
        Param_temp[i_dummy] = Param_Data[Row, 1 .+ i_dummy]
    end

    (VarName_i, x_i) = Func_if_VarName(num_param_i)
    @show VarName_i

    LogL_range = zeros(length(x_i))
    for x_ind_i = 1:length(x_i)
        Param_temp[num_param_i] = x_i[x_ind_i]

        Param1.p_change = Param_temp[1]
        Param1.ϵ = Param_temp[2]
        Param1.β_R = Param_temp[3]
        Param1.β_ratio_N2R = Param_temp[4]
        Param1.γ_RL_R = Param_temp[6]
        Param1.γ_RL_N = Param_temp[7]
        Param1.λ_elig_R = Param_temp[8]
        Param1.λ_elig_N = Param_temp[9]
        Param1.α = Param_temp[10]
        Param1.Δα = Param_temp[11]
        Param1.Q_N_0 = Param_temp[12]
        Param1.N_prior_sweep = Param_temp[13]
        Param1.w_MF_to_MB_2 = Param_temp[14]
        Param1.w_MF_to_MB_1 = Param_temp[15]
        Param1.w_MF_to_MB_Scaler = Param_temp[17]

        Param2 = deepcopy(Param1)
        Param2.β_ratio_N2R = Param_temp[5]
        Param2.w_MF_to_MB_1 = Param_temp[16]
        Param2.β_R = Param_temp[18]

        Param = [Param1, Param2]

        (Agents, LogL) = SurNoR_2020.Func_train_set_of_Agents(BehavData;
                                                               Sub_Set = Sub_Set,
                                                               Epi_Set=Epi_Set,
                                                               Block_Set=Block_Set,
                                                               Param=deepcopy(Param),
                                                               HyperParam=HyperParam)
        LogL_range[x_ind_i] = LogL
    end

    A = DataFrame(x=Array(x_i),y=LogL_range)
    Path_Save = "data/LogL_Landscape/"
    CSV.write(string(Path_Save, "LogL_range_i", string(num_param_i),"_Row", string(Row), ".CSV" ),A)
end
