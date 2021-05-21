# The code for organizing the optimization results for all  4 repetitions of
# 3-fold cross-validation together and saving them in data
using CSV
using DataFrames
using Query
using PyPlot
using Statistics

# ------------------------------------------------------------------------------
# Defining Hyper Parameters
Path_Main = "src/Optimization/CV_3Fold/"

Updates =  ["MB_Perfect", "MB_Leaky", "MBQ0S","MBS",
            "MFQ0", "MFN", "MFNS",
            "HybQ0S","HybN", "SurNoR",
            "MBSExp","MFSExp","HybSExp",
            "HybSTrapDet","HybSTrapDet2","SurNoR_NovInExploit"]

# ------------------------------------------------------------------------------
# Load Data
Chains = 100
for Alg_ind = 1:length(Alg_Path)
    Path_Files = Path_Main
    update_form = Updates[Alg_ind]
    for Fold = 1:3
        @show Fold
        Chain = 1
        Path = string(Path_Files, "CoAscData_", update_form, "_F", string(Fold), "_C",
                       string(Chain), ".csv")

        CoAsc_Data = CSV.read(Path)

        for Chain = 2:Chains
            Path = string(Path_Files, "CoAscData_", update_form, "_F", string(Fold), "_C",
                           string(Chain), ".csv")
            if isfile(Path)
                temp = CSV.read(Path)
                CoAsc_Data = cat(CoAsc_Data,temp; dims=1)
            end
        end


        # ------------------------------------------------------------------------------
        # Plot Log Series
        fig = figure(); ax = gca()
        ax.plot(CoAsc_Data[1].LogL[CoAsc_Data[1].LogL .!= 0], linewidth=0.1)
        for i = 2:length(CoAsc_Data)
            @show i
           ax.plot(CoAsc_Data[i].LogL[CoAsc_Data[i].LogL .!= 0], linewidth=0.1)
        end
        title(string(update_form, " - LogL Series, Fold ", string(Fold) ))
        ax.grid(true)
        Path_save = "data/Fitted_parameters/CV_3Fold/"
        savefig(string(Path_save, update_form, " - LogL Series, Fold ", string(Fold), ".pdf" ))
        close(fig)
        # ------------------------------------------------------------------------------
        # V_Cat
        CoAsc_Data_Series = CoAsc_Data[1]

        for i = 2:length(CoAsc_Data)
            CoAsc_Data_Series = vcat(CoAsc_Data_Series, CoAsc_Data[i])
        end
        CoAsc_Data_Series = @from i in CoAsc_Data_Series begin
                           @where i.LogL < 0
                           @select i
                           @collect DataFrame
        end
        unique!(CoAsc_Data_Series)

        BestLogL = findmax(CoAsc_Data_Series.LogL)
        BestParam = CoAsc_Data_Series[BestLogL[2],:]
        if Fold==1
            global BestLogL_Folds = BestLogL[1]
            global BestParam_Folds = BestParam
        else
            BestLogL_Folds = cat(BestLogL_Folds, BestLogL[1], dims=1)
            BestParam_Folds = vcat(BestParam_Folds, BestParam[:])
        end
    end

    BestParam_Save = zeros((3,(length(BestParam_Folds[1])+1)))
    for Fold = 1:3
        BestParam_Save[Fold, 1] = Fold
        for i = 1:length(BestParam_Folds[1])
            BestParam_Save[Fold, i+1] = BestParam_Folds[Fold][i]
        end
    end

    BestParam_Save = DataFrame(BestParam_Save)
    rename!(BestParam_Save, vcat(["Fold"],names(BestParam_Folds[1])))

    Path = string(Path_save, update_form, "_BestParams.csv" )
    CSV.write(Path, BestParam_Save)
end
