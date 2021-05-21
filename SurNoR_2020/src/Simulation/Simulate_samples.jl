# The code to simulate 200 artificial participants given the SurNoR parameters
# fitted to behavior
using SurNoR_2020
using MAT
using PyPlot
using CSV
using JLD

i = 3
# --------------------------------------------------------------------------
# Algorithm
update_form = "VarSMiLe"

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
HyperParam = SurNoR_2020.Str_HyperParam(; gamma_form=gamma_form,
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
Path_Params = "data/Fitted_parameters/Overall/OverAll_SurNoR_BestParams_diff5.csv"
Param_Data = CSV.read(Path_Params)

η0 = Array(Param_Data[i,2:end])
σ0 = [0.01,1e-5,2,0.1,0.1,0.01,0.1,0.01,0.1,0.1,0.1,1,5,0.1,0.1,0.1,1,2]
noise_free = true

# ------------------------------------------------------------------------------
# Loading the environment
Path = string("data/HardEnvironment11s.mat")
EnvGraph_MAT = matread(Path)
EnvGraph1 = SurNoR_2020.Str_EnvGraph(; TranMat = EnvGraph_MAT["Hard_NoSwitch"]["tm"].-1,
                                            InitObs = EnvGraph_MAT["Hard_NoSwitch"]["initS"].-1,
                                            Goal=0)

EnvGraph2 = SurNoR_2020.Str_EnvGraph(; TranMat = EnvGraph_MAT["Hard_Switch411"]["tm"].-1,
                                            InitObs = EnvGraph_MAT["Hard_Switch411"]["initS"].-1,
                                            Goal=0)

InitObs = [9,6,8,4,2]
for i = 1:5
  EnvGraph1.InitObs[i] = InitObs[i]
  EnvGraph2.InitObs[i] = InitObs[i]
end

EnvGraph_Array = [EnvGraph1, EnvGraph2]

# ------------------------------------------------------------------------------
# Simulate
N = 200
Seed_start = 2021
N_lost = [0]
SimAgent = []
for n = 1:N
  println("----------------------------------------")
  @show n
  @show Seed_start + n
  Param = SurNoR_2020.Func_param_generator(η0;
                                           σ = σ0./10,
                                           noise_free=noise_free,
                                           Seed= Seed_start + n)
  Temp_Age = SurNoR_2020.Func_simulate_Agent_Blocks(EnvGraph_Array;
                                                  Epi_Set=1:5, Block_Set=1:2,
                                                  Param=Param,
                                                  HyperParam=HyperParam,
                                                  Greedy = false,
                                                  Seed= Seed_start + n,
                                                  T_stop = 500)
  if size(Temp_Age) == (2,5)
    @show length(Temp_Age[1,1].Input.act')
    @show length(Temp_Age[2,1].Input.act')
    push!(SimAgent,Temp_Age)
  else
    N_lost[1] = N_lost[1]+1
  end
end

N = N - N_lost[1]
Inputs = Array{SurNoR_2020.Str_Input,3}(undef,N,2,5)
for n = 1:N
  for Block = 1:2
    for Epi = 1:5
      Inputs[n,Block,Epi] = SimAgent[n][Block,Epi].Input
    end
  end
end
save("data/Simulated_data/Simulated_data.jld", "Inputs", Inputs, "N_lost", N_lost)
