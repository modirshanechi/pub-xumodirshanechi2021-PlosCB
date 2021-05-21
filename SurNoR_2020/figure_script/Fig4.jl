# The code to generate Fig 4
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
# Fig 4A: Surprise Time Series
# Only the data for subject 7 is shown in the paper
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Epi = 1
Block = 2
for Sub = 1:12
    Agent = Agents_SurNoR[Sub,Block,Epi]

    fig = figure(figsize=(7,6)); ax = gca()
    ax.set_xlabel("Time")
    ax.set_ylabel("Surprise")
    ax.plot(Agent.Surprise[2:end])
    ax.set_ylim([0.5,1.5e5])
    ax.set_yscale("log")
    title(string("Surprise over Time, 2nd Block, 1st Epi., Sub. ",string(Agent.Sub)))
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Fig 4B: Surprise HeatMap
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
Epi = 1
Block = 2
Surp_temp = []
# All pairs of state and actions
States = 0:43
for s = States
    push!(Surp_temp,[])
end

# Gathering the maximum surprise values for all pairs of state and actions
for Sub = 1:12
    Agent = Agents_SurNoR[Sub,Block,Epi]
    for s = States
        Ind = 1:length(Agent.State)
        if sum(Agent.State.==s)>0
            Ind = Ind[Agent.State.==s] .+ 1
            append!(Surp_temp[s+1], log10.(findmax(Agent.Surprise[Ind])[1]))
        end
    end
end

for s = States
    Surp_temp[s+1] = Float64.(Surp_temp[s+1]')
end

# Putting the values in a matrix
Surp_Matrix = zeros(10,4)
Map = [1,5,10,8,4,9,3,2,6,7] .+ 1
for s = 1:10
    for a = 1:4
        Surp_Matrix[s,a] = mean(Surp_temp[(Map[s]-1)*4 + a])
    end
end

# Plotting
fig = figure(figsize=(6,6)); ax = gca()
sc = ax.imshow(Surp_Matrix,vmin=0,vmax=5)

ax.set_yticks(0:10)
ax.set_yticklabels(["State 1", "State 2", "State 7 (S)",
                     "State 4", "State 5", "State 6", "State 3 (S)",
                     "State 8", "State 9", "State 10"],rotation=0)

ax.set_xticks(0:3)
ax.set_xticklabels(["A1", "A2", "A3", "A4"])

ax.set_ylim([9.5,-0.5])
ax.set_xlim([-0.5,3.5])

title("Heatmap Average Maximum log Surprise-Value in the 1st Episode of 2nd Block")

cbar= fig.colorbar(sc)
cbar.set_label("Surprise Value")

fig.subplots_adjust(bottom = 0.1, left = -0.3)
