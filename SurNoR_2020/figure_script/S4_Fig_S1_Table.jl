# The code to generate S4 Fig and Data for S1 Table
using PyPlot
using SurNoR_2020
using Statistics
using CSV
using DataFrames

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

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

# 3 sets of recovered parameters
N_set = [1,2,3]

# Loading fitted parameters
Path_Params = "data/Fitted_parameters/Overall/"
Path_Params = string(Path_Params, "OverAll_SurNoR_BestParams_diff5.csv" )
Param_Data = CSV.read(Path_Params)
Row = 3

# Loading recovered parameters
Param_Data_nset_all = []
for n_set = N_set
    Path_Params_nset = "data/Fitted_parameters_simulated/Overall/"
    Path_Params_nset = string(Path_Params_nset, "nset_", string(n_set), "/", "OverAll_SurNoR_BestParams_diff5.csv" )
    Param_Data_nset = CSV.read(Path_Params_nset)
    push!(Param_Data_nset_all,Param_Data_nset)
end
Row_nset = [1,1,1]

# Parameter structure
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

# An array for rrrors in S1 Table
Sigma = zeros(size(Param_Data[Row,:]))

# Plotting panels of S4 Fig
Param_perm_order = 1:18
for i = 1:length(Param_perm_order)
    Path_Load = "data/LogL_Landscape/"
    num_param_i = Param_perm_order[i]
    (VarName_i, x_i) = Func_if_VarName(num_param_i)

    Path = string(Path_Load, "LogL_range_i", string(num_param_i), "_Row", string(Row), ".CSV" )

    @show VarName_i

    LogL_range = CSV.read(Path)
    LogL_range = Matrix(LogL_range)[:,2]

    LogL_range[isnan.(LogL_range)] .= -1e6
    LogL_range[isinf.(LogL_range)] .= -1e6

    y = LogL_range
    y_0 = findmax(y)[1]
    x_0_ind = findmax(y)[2]
    x_0 = x_i[x_0_ind]

    dx_res = [0.]
    # Laplace approximation
    if x_0_ind == 1
        dy = abs(y[x_0_ind+1] - y[x_0_ind])
        dx = abs(x_i[x_0_ind+1] - x_i[x_0_ind])
        c = 2 * dy / (dx^2)
        dx_res[1] = dx
    elseif x_0_ind == length(x_i)
        dy = abs(y[x_0_ind] - y[x_0_ind-1])
        dx = abs(x_i[x_0_ind] - x_i[x_0_ind-1])
        c = 2 * dy / (dx^2)
        dx_res[1] = dx
    else
        dy1 = abs(y[x_0_ind] - y[x_0_ind-1])
        dy2 = abs(y[x_0_ind+1] - y[x_0_ind])
        dx1 = abs(x_i[x_0_ind] - x_i[x_0_ind-1])
        dx2 = abs(x_i[x_0_ind+1] - x_i[x_0_ind])
        c = (dy1/dx1 + dy2/dx2) / (dx1 + dx2) * 2
        dx_res[1] = max(dx1,dx2)
    end
    Sigma[i+1] = max(1 / sqrt(c), dx_res[1])

    X_0_nset = []
    for n_set = N_set
        Param_Data_nset = Param_Data_nset_all[n_set]
        x_0_nset = Param_Data_nset[Row_nset[n_set],1+i]
        append!(X_0_nset,x_0_nset)
    end

    y = LogL_range
    y_max = findmax(y)[1]
    y_min = findmin(y)[1]

    fig = figure(); ax = gca()
    ax.plot(x_i,y,color="k")
    ax.plot([x_0,x_0],[y_min,y_max],color="#487952",alpha = 0.5)
    for n_set = N_set
        x_0_nset = X_0_nset[n_set]
        ax.plot([x_0_nset,x_0_nset],[y_min,y_max],color="#7B082E",alpha = 0.5)
    end
    title(string("SurNoR - Loss Shape - ", VarName_i))
    if VarName_i == "eps"
            xscale("log")
    end
end

# Saving data for S1 Table
A = push!(DataFrame(Param_Data[Row,:]), round.(Sigma, digits = 5))
Path_Save = "data/LogL_Landscape/"
CSV.write(string(Path_Save, "SurNoR_Params_Sigma_rounded.CSV" ),A)
