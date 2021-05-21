# The code to generate Fig 3
using PyPlot
using SurNoR_2020
using Statistics

PyPlot.svg(true)
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42

# ------------------------------------------------------------------------------
# Training Algorithms
# ------------------------------------------------------------------------------
Agents_SurNoR = SurNoR_2020.Func_SurNoR_Overall_train_opt_ind_i(3)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 3A: Count and Novelty Time Series
# Only the data for subject 9 is shown in the paper
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Epi = 1
Block = 1
Colors = ["#E56969","#C14545","#B71616","#45A2C1","#167CB7","#B159E0"]
for Sub = 1:12
    Agent = Agents_SurNoR[Sub,Block,Epi]

    State_Set = [9,10,11,5,8,1]
    Order = [1,2,6,4,9,5,10,11,3,7,8]
    Names = ["State 8 (Trap)", "State 9 (Trap)", "State 10 (Trap)", "State 4", "State 7", "Goal State"]

    fig = figure(figsize=(7,6)); ax = gca()
    for j = 1:length(State_Set)
       i = State_Set[j]
       ax.plot(Agent.Counter[Order[i],:],color=Colors[j])
    end
    title(string("Counter over Time, 1st Block, 1st Epi., Sub. ",string(Agent.Sub)))
    ax.legend(Names)
    ax.set_xlabel("Time")
    ax.set_ylabel("Counter");

    fig = figure(figsize=(7,6)); ax = gca()
    for j = 1:length(State_Set)
       i = State_Set[j]
       ax.plot(Agent.Novelty[Order[i],:],color=Colors[j])
    end
    title(string("Novelty over Time, 1st Block, 1st Epi., Sub. ",string(Agent.Sub)))
    ax.legend(Names)
    ax.set_xlabel("Time")
    ax.set_ylabel("Novelty");
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 3B: Novelty Map
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Novelty = zeros(11,12)
Epi = 1
Block = 1
for Sub = 1:12
    Ind = 1:length(Agents_SurNoR[Sub,Block,Epi].Input.obs)
    Ind = Ind[Agents_SurNoR[Sub,Block,Epi].Input.obs .== 10]
    Novelty[:,Sub] = Agents_SurNoR[Sub,Block,Epi].Novelty[:,findmin(Ind)[1]]
end

Novelty_Mean = mean(Novelty,dims=2)

Order = [1,5,3,8,4,9,10,0,2,6,7]
Names = ["1","2","3","4","5","6","7","G","8","9","10"]

N_Ordered = zeros(size(Novelty_Mean))
for i=1:length(Novelty_Mean)
    N_Ordered[i] = Novelty_Mean[Order[i]+1]
end

x = [1,2,3,4,5,6,7,8, 3.5, 4.5, 5.5]
y = [1,1,1,1,1,1,1,1, 0.7, 0.7, 0.7]

fig = figure(figsize=(8,6)); ax = gca()
sc = ax.scatter(x, y, s=ones(size(x)).*1200, c=N_Ordered[:])

for i =1:length(Names)
 ax.text(x[i],y[i],Names[i],fontsize=12,
          horizontalalignment="center", verticalalignment="center")
end
ax.grid(false)
ax.set_xticklabels([]); ax.set_yticklabels([])
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-0.1,9.1); ax.set_ylim(0.2,1.4)
ax.set_title("Average of Novelty of States - 1st Episode of the 1st Block")

cbar= fig.colorbar(sc)
cbar.set_label("Novelty Value")
