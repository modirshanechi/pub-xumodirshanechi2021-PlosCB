using CSV
using DataFrames
using Query
using PyPlot
using Statistics

# Different choices of n_set is needed for different seeds (1, 2, or 3)
n_set = 1
# ------------------------------------------------------------------------------
# Defining Hyper Parameters
update_form = "SurNoR"
Path_Main = "src/Optimization_Simulated_Data/OverAll/"
Path_Files = Path_Main

# ------------------------------------------------------------------------------
# Load Data
Chains = 500

Chain = 1
Path = string(Path_Files, "CoAscData_OverAll_SurNoR_nset_", string(n_set), "_C",
               string(Chain), ".csv")

global CoAsc_Data = CSV.read(Path)

for Chain = 2:Chains
    Path = string(Path_Files, "CoAscData_OverAll_SurNoR_nset_", string(n_set), "_C",
                   string(Chain), ".csv")
    if isfile(Path)
        temp = CSV.read(Path)
        global CoAsc_Data = cat(CoAsc_Data,temp; dims=1)
    end
    @show Chain
end

for Chain = 1:length(CoAsc_Data)
    CoAsc_Data[Chain] = CoAsc_Data[Chain][CoAsc_Data[Chain].LogL .< 0,:]
end

# ------------------------------------------------------------------------------
# Plot Log Series
fig = figure(); ax = gca()
ax.plot(CoAsc_Data[1].LogL, linewidth=0.1)
for i = 2:length(CoAsc_Data)
   ax.plot(CoAsc_Data[i].LogL, linewidth=0.1)
end
title(string(update_form, " - LogL Series" ))
ax.grid(true); ax.legend()
Path_save = "data/Fitted_parameters_simulated/OverAll/"
savefig(string(Path_save, update_form, " - LogL Series_nset_", string(n_set), ".pdf" ))
close(fig)
# ------------------------------------------------------------------------------
# V_Cat
global CoAsc_Data_Series = CoAsc_Data[1][end,:]
for i = 2:length(CoAsc_Data)
    global CoAsc_Data_Series = vcat(CoAsc_Data_Series, CoAsc_Data[i][end,:])
end
CoAsc_Data_Series = @from i in CoAsc_Data_Series begin
                   @where i.LogL < 0
                   @select i
                   @collect DataFrame
                   end
unique!(CoAsc_Data_Series)

global BestLogL = findmax(CoAsc_Data_Series.LogL)
global BestParam = CoAsc_Data_Series[BestLogL[2],:]

Ind = CoAsc_Data_Series.LogL .> (BestLogL[1]-5)
global BestParam_diff5 = CoAsc_Data_Series[Ind,:]

BestParam_Save = DataFrame(BestParam)

Path = string(Path_save, "nset_", string(n_set), "/", "OverAll_", update_form, "_BestParams.csv")
CSV.write(Path, BestParam_Save)

BestParam_diff5_Save = DataFrame(BestParam_diff5)
sort!(BestParam_diff5_Save, :LogL, rev=true)
Path = string(Path_save, "nset_", string(n_set), "/", "OverAll_", update_form, "_BestParams_diff5.csv")
CSV.write(Path, BestParam_diff5_Save)
