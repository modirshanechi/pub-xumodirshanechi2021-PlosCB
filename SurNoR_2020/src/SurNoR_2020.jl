module SurNoR_2020

using CSVFiles
using CSV
using DataFrames
using DataFramesMeta
using Statistics
using SpecialFunctions
using Random
using PyPlot
using HypothesisTests

include("Structs_form.jl")
include("Functions_for_model.jl")
include("Functions_for_data.jl")
include("Functions_for_behavioral_landmarks.jl")
include("Functions_for_simulation.jl")
include("Functions_for_EEG.jl")
include("Functions_for_model_evaluation.jl")

end # module
