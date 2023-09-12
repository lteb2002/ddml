


push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./rere_dml")
push!(LOAD_PATH, "./lpsolver")

using CSV
using Tables
using DataFrames
using LinearAlgebra
using StatsBase
using Statistics
using DiagDml
using Distributed
using TripletModule


path="G:\\dataset\\dml_feature_selection_data\\"
f = "credit_score2_samples"
# f="wine"


path3 = path*f*"\\coverge_test\\"

fn=path*f*".csv"
println("Number of workers:",nworkers())
println("Loading data from $fn--------------------------")
csv_data = CSV.read(fn,DataFrame)
data = Array{Float32,2}(csv_data[:,1:end-1])
dt = fit(UnitRangeTransform, data, dims=1)
# dt = fit(ZScoreTransform, data, dims=1)
data = StatsBase.transform(dt, data)
labels = "label_".*string.(csv_data[:,end])
# println(labels)
# println(labels)

ts=-4:-1.0:-10
distance_type = "huber2"
triplets = TripletModule.build_triplets(data, labels)
for t in ts
    tau = 2^t
    println("Current task:",t)
    # k=DiagDml.test_lipschitz()
    # println("Current Lipschitz K:",k)
    try
        @time x=DiagDml.solve_diag_dml(triplets,1.0,0.5,distance_type,tau)
        println("Solutions of t:"*string(t)*":",x)
        new_data = data * Diagonal(x)
        # println(new_data)
        csv = hcat(new_data,labels)
        # println(csv)
        mkpath(path3)
        output = path3*f*"_t_"*string(t)*".csv"
        table = Tables.table(csv)
        CSV.write(output,table)
        println("")
    catch e
        println(e)
    end
end

# cor0 = cor(w_stack,dims=1)
# println(cor0)
#
# using Plots
# plotly()
# plot(w_stack)
# heatmap(1:size(cor0,1),
#     1:size(cor0,2), cor0,
#     c=cgrad([:white, :yellow,:orange,:red]),
#     xlabel="x values", ylabel="y values",
#     title="My title")
