# The code to generate Fig 1A
using PyPlot
using CSV
using SurNoR_2020
using Statistics
using HypothesisTests

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# ------------------------------------------------------------------------------
# Behavioral data
# ------------------------------------------------------------------------------
BehavData = CSV.read("./data/BehavData_SwitchState.csv")

Sub_Num = 12
Sub_Set = [1,2,3,4,5,6,7,8,9,11,12,13]
Inputs = Array{SurNoR_2020.Str_Input,3}(undef,Sub_Num,2,5)
for n = 1:Sub_Num
  for Block = 1:2
    for Epi = 1:5
      Inputs[n,Block,Epi] = SurNoR_2020.Func_data_to_input(BehavData; Sub=Sub_Set[n],
                                                           Epi=Epi, Block=Block)
    end
  end
end

# ------------------------------------------------------------------------------
# Lenght of episodes
# ------------------------------------------------------------------------------
Len_Mean, Len_Std, Len_Matrix = SurNoR_2020.Func_episode_lenght(Inputs)

println("The t-test result for comparing average number of actiopns in B1E1 and B2E2:")
@show OneSampleTTest(Len_Matrix[:,1,1], Len_Matrix[:,1,2])

fig = figure(); ax = gca()

x = ((1:1:5))
y=Len_Mean[:,:,1]
dy = Len_Std[:,:,1]/sqrt(length(Len_Matrix[:,1,1]))
bp1 = ax.bar(x, y[:] , yerr=dy[:], color="#167CB7")

x = ((6:1:10))
y=Len_Mean[:,:,2]
dy = Len_Std[:,:,2]/sqrt(length(Len_Matrix[:,1,1]))
bp2 = ax.bar(x, y[:] , yerr=dy[:], color="#B71616")
ax.set_xticks(1:1:10)
ax.set_xticklabels(["E1","E2","E3","E4","E5","E1","E2","E3","E4","E5"])
legend(["Block 1","Block 2"])

x = Array(1:1:10)
σ = 0.1
for j = 1:Sub_Num
  x_1 = x .+ 2*σ*(rand() - 0.5)
  y = Len_Matrix[j,:,:]
  ax.plot(x_1, y[:],".",color="k",alpha=0.5)
end
ax.set_ylim([0,250])

ax.set_title("Number of Steps of Subjects")
ax.set_xlabel("Episode")
ax.set_ylabel("Number of Steps")
