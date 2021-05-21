# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing episode lenghts
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_episode_lenght(Inputs; inds = -1)
    if inds != -1
        Inputs = Inputs[inds,:,:]
    end
    Sub_Num = size(Inputs)[1]
    Len_Matrix = zeros(Sub_Num,5,2)

    for Sub = 1:Sub_Num
        for Epi = 1:5
            for Block = 1:2
                Len_Matrix[Sub,Epi,Block] = length(Inputs[Sub,Block,Epi].obs)
            end
        end
    end

    Len_Mean = mean(Len_Matrix, dims=1)
    Len_Std = std(Len_Matrix, dims=1)

    return Len_Mean, Len_Std, Len_Matrix
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing trap avoidance
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_trap_av_lenght(Inputs; inds = -1, Block = 1, Epi = 1,
                                     Trap_set = [2,6,7], K_movemean = 1)
    if inds != -1
        Inputs = Inputs[inds,:,:]
    end
    Sub_Num = size(Inputs)[1]

    Lenghts = Array{Array{Float64,1},1}(undef,Sub_Num)

    for Sub = 1:Sub_Num
      Obs = Inputs[Sub,Block,Epi].obs
      istrapObs = zeros(size(Obs))
      for i=1:length(Obs)
        istrapObs[i] = (Obs[i] ∈ Trap_set)
      end
      n = length(Obs)
      Inds = Array(2:n)
      y = (diff(istrapObs) .+ 1) ./ 2
      Lenghts[Sub] = Inds[y .== 0] .- Inds[y .== 1]
      Lenghts[Sub] = Func_movmean(Lenghts[Sub],K_movemean)
    end

    x_len = sort(length.(Lenghts),rev=true)[2]
    y = zeros(x_len)
    dy = zeros(x_len)
    y_med = zeros(x_len)
    y_Q25 = zeros(x_len)
    y_Q75 = zeros(x_len)
    N = zeros(x_len)
    for i = 1:x_len
        temp = []
        for Sub = 1:Sub_Num
            if length(Lenghts[Sub]) >= i
                push!(temp, Lenghts[Sub][i])
            end
        end
        y[i] = mean(temp)
        y_med[i] = median(temp)
        y_Q25[i] = quantile(temp, 0.25)
        y_Q75[i] = quantile(temp, 0.75)
        N[i] = length(temp)
        dy[i] = std(temp) / sqrt(length(temp))
    end

    return y, dy, N, Lenghts, y_med, y_Q25, y_Q75
end


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Computing average progress
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_action_progress_vec(Inputs, AC; inds = -1, Block = 1, Epi = 1,
                                              State = 5, only_prog=false,
                                              K_movemean = 0, scores = [-1,0,1])
    if inds != -1
        Inputs = Inputs[inds,:,:]
    end
    Sub_Num = size(Inputs)[1]

    if only_prog
        scores = [0,0,1]
    end

    Actions = Array{Array{Float64,1},1}(undef,Sub_Num)

    for Sub = 1:Sub_Num
      Obs = Inputs[Sub,Block,Epi].obs
      Act = Inputs[Sub,Block,Epi].act
      Act = Act[Obs .== State]

      n = length(Act)
      if n>0
          Actions[Sub] = ones(n) .* scores[1]
          Actions[Sub][Act .== AC[State+1,2]] .= scores[2]
          Actions[Sub][Act .== AC[State+1,1]] .= scores[3]
          Actions[Sub] = Func_movmean(Actions[Sub],K_movemean)
      else
          Actions[Sub] = deepcopy(Act)
      end
    end

    x_len = sort(length.(Actions),rev=true)[2]
    y = zeros(x_len)
    dy = zeros(x_len)
    y_med = zeros(x_len)
    N = zeros(x_len)
    p_values = zeros(x_len)
    for i = 1:x_len
        temp = []
        for Sub = 1:Sub_Num
            if length(Actions[Sub]) >= i
                push!(temp, Actions[Sub][i])
            end
        end
        y[i] = mean(temp)
        y_med[i] = median(temp)
        N[i] = length(temp)
        dy[i] = std(temp) / sqrt(length(temp))
        p_values[i] = pvalue(OneSampleTTest(Float64.(temp)))
    end

    return y, dy, N, Actions, y_med, p_values
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Moving Average
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
function Func_movmean(x,k;ZeroPadding = false)
    n = length(x)
    y = zeros(n)
    if ZeroPadding
        x = append!(zeros(k),x)
        x = append!(x,zeros(k))
        inds = 0:(2*k)
        for i = 1:n
            y[i] = mean(x[i .+ inds])
        end
    else
        for i = 1:n
            if i <= k
                ind_beg = 1
            else
                ind_beg = i - k
            end
            if n <= i+k
                ind_end = n
            else
                ind_end = i + k
            end
            y[i] = mean(x[ind_beg:ind_end])
        end
    end
    return y
end
export Func_movmean
