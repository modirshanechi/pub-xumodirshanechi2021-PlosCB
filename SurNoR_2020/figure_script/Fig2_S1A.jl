# The code to generate Fig 2 and SFig 1A
using PyPlot
using CSV
using SurNoR_2020
using Statistics
using MAT

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
# Loading the action classes
# ------------------------------------------------------------------------------
ActionClass = matread("data/ActionClasses.mat")
AC1 = ActionClass["A1"]
AC2 = ActionClass["A2"]

# ------------------------------------------------------------------------------
# Fig 2A Trap-avoidance lenght
# ------------------------------------------------------------------------------
Epi = 1
K_movemean=1
N_thresh = 2

Trap_set = [1,2,6,7]

for Block = 1:2
  y, dy, N, Lenghts, y_med, y_Q25, y_Q75 = SurNoR_2020.Func_trap_av_lenght(Inputs;
                          Trap_set = Trap_set, Block=Block,K_movemean=K_movemean)

  if Block == 1
    Color = "#167CB7"
  else
    Color = "#B71616"
  end

  fig = figure(); ax = gca()
  ax.plot(N)
  ax.set_xlabel("Number of visit")
  ax.set_ylabel("Number of subjects")

  fig = figure(); ax = gca()

  x = 1:length(N)
  x = x[N .>= N_thresh]
  x_lim = length(x)

  dy_med = zeros(length(x),2)
  dy_med[:,1] .= y_med[x] .- y_Q25[x]
  dy_med[:,2] .= y_Q75[x] .- y_med[x]

  ax.errorbar(x,y_med[x],yerr = dy_med',color=Color,capsize=3)
  ax.scatter(x,y_med[x],s=10 .* N,c=Color)
  ax.set_xlim([0, x_lim + 1])

  ax.set_title("Number of actions to escape traps at Nth visit")
  ax.set_xlabel("Number of visit")
  ax.set_ylabel("Number of actions")

  ax.plot(-1:50, ones(52).*2, "--k",alpha=0.5)
  σ = 0.1
  for j = 1:Sub_Num
    y_new = Lenghts[j]
    if length(y_new) > x_lim
      y_new = y_new[1:x_lim]
    end
    x_1 = (1:length(y_new)) .+ 2*σ*(rand() - 0.5)
    ax.plot(x_1, y_new[:],".",color="k",alpha=0.5)
  end

  ax.set_ylim([1,18])
end

# ------------------------------------------------------------------------------
# Fig 2B, Fig 2C, and SFig 1A: Progress measures
# ------------------------------------------------------------------------------
Epi = 1
K_movemean = 1
scores = [-0.75,0.5,1.]
Map_S = [1,8,3,5,2,9,10,4,6,7]

for Block = 1:2
  fig = figure(figsize=(28,28))
  for State = 1:10
    if Block == 1
      AC = AC1
      Color = "#167CB7"
    else
      AC = AC2
      Color = "#B71616"
    end
    y, dy, N, Actions, y_med = SurNoR_2020.Func_action_progress_vec(Inputs, AC;
                                                        Block = Block, Epi = Epi,
                                                        State = State,
                                                        K_movemean=K_movemean,
                                                        scores=scores)

    N_thresh = 2
    ax = subplot(3,4,Map_S[State])

    x = 1:length(N)
    x = x[N .>= N_thresh]

    ax.errorbar(x,y[x],yerr = dy[x],color=Color,capsize=3)
    ax.scatter(x,y[x],s=10 .* N,c=Color)
    ax.set_ylim([-1.1,1.1])


    x_lim = length(x)
    ax.plot(-1:(x_lim+1), -ones(x_lim+3).*1.0, "--k",alpha=0.5)
    ax.plot(-1:(x_lim+1), ones(x_lim+3).*1.0, "--k",alpha=0.5)
    ax.plot(-1:(x_lim+1), ones(x_lim+3).*0.0, "--k",alpha=0.5)

    ax.set_title(string("Average progress at Nth visit - State:", string(Map_S[State])))
    ax.set_ylabel("Average progress")
    ax.set_xlim([0, x_lim + 1])
  end
end
