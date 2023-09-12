


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
using TripletModule
using Distributed


path="G:\\dataset\\dml_feature_selection_data\\"
f = "credit_score2"

path2 = path*f*"\\"
path3 = path*f*"\\reg_path\\"

fn=path*f*".csv"
println("Number of workers:",nworkers())
println("Loading data from $fn--------------------------")
csv_data = CSV.read(fn,DataFrame)
data = Array{Float32,2}(csv_data[:,1:end-1])
# dt = fit(ZScoreTransform, data, dims=1)
dt = fit(UnitRangeTransform, data, dims=1)
data = StatsBase.transform(dt, data)
labels = "label_".*string.(csv_data[:,end])
# println(labels)
# println(labels)

lns = 10.0 : -0.5: 5.5
as=1.0:0.1:1.0
distance_type = "huber2"
triplets = TripletModule.build_triplets(data, labels)
for a in as
    println("Current task:",a)
    w_stack = zeros(length(lns),size(data)[2])
    Threads.@threads for (i,pow) in collect(pairs(lns))
        w = 10.0^pow
        @time x=DiagDml.solve_diag_dml(triplets,w,a,distance_type)
        # @time x=DiagDml.solve_diag_dml(data,labels,w,a,distance_type)
        println("Solutions of a:"*string(a)*",w:"*string(pow)*":",x)
        new_data = data * Diagonal(x)
        # println(new_data)
        csv = hcat(new_data,labels)
        # println(csv)
        mkpath(path2)
        output = path2*f*"_a_"*string(a)*"_w_"*string(pow)*".csv"
        table = Tables.table(csv)
        CSV.write(output,table)
        w_stack[i,:] = x
    end
    println(w_stack)
    mkpath(path3)
    output = path3*f*"_a_"*string(a)*"_path.csv"
    table = Tables.table(w_stack)
    CSV.write(output,table)
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
