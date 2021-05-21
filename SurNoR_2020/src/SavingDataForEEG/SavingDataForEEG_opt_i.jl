# A code to save the modeling data in the MATLAB structure to be used for EEG
# analysis
using SurNoR_2020
using DataFrames
using MAT

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

for opt_i = 3
    # Training Algorithms
    Agents_SurNoR   = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(opt_i)
    Agents_Tot_Alg = cat(Agents_SurNoR, dims=4)
    Path_save = "data/ModelData_for_EEG/"

    # MATLAB Version of struct
    for Sub = 1:12
        for Block = 1:2
            for Epi = 1:5
                if Epi==1

                    global Agents_Epi_MATLAB = SurNoR_2020.Str_Agent_MAT(Agent=deepcopy(Agents_SurNoR[Sub,Block,Epi]))
                else
                    Agents_Epi_MATLAB = cat(Agents_Epi_MATLAB,
                                            SurNoR_2020.Str_Agent_MAT(Agent=deepcopy(Agents_SurNoR[Sub,Block,Epi])),
                                            dims=3)
                end
            end
            if Block==1
                global Agents_Block_MATLAB = Agents_Epi_MATLAB
            else
                Agents_Block_MATLAB = cat(Agents_Block_MATLAB,
                                          Agents_Epi_MATLAB, dims=2)
            end
        end
        if Sub==1
            global Agents_Sub_MATLAB = Agents_Block_MATLAB
        else
            Agents_Sub_MATLAB = cat(Agents_Sub_MATLAB,
                                    Agents_Block_MATLAB, dims=1)
        end
    end

    Path = string(Path_save,"Trained_Data", string(opt_i) ,".mat" )
    file = matopen(Path, "w")
    write(file, "Data", Agents_Sub_MATLAB)
    close(file)
end
