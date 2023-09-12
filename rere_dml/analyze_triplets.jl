push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./rere_dml")
push!(LOAD_PATH, "./lpsolver")

using CSV
using Tables
using DataFrames
using LinearAlgebra
using StatsBase
using DiagDml
using TripletModule
using Statistics


path="E:\\julia_work\\test\\data\\"
f = "diabetes"

fn=path*f*".csv"

println("Loading data from $fn--------------------------")
csv = CSV.read(fn,DataFrame)
data = csv[:,1:end-1]
# data = convert(Array{Float32,2}, csv[:,1:end-1])
data = Array{Float32,2}(data)
dt = fit(ZScoreTransform, data, dims=1)
data = StatsBase.transform(dt, data)
# println(data)
labels = "label_".*string.(csv[:,end])
# println(labels)
triplets = TripletModule.build_triplets(data, labels)
if true
    regWeight = 10
    @time x=DiagDml.solve_diag_dml(data,labels,regWeight,1)
    println("Solutions:",x)
    for t in triplets
        t.xi = t.xi .* x
        t.xj = t.xj .* x
        t.xk = t.xk .* x
        t.jk_dis = norm(t.xj - t.xk)
        t.ij_dis = norm(t.xj - t.xi)
    end
    # println(new_data)
end

diss = [t.jk_dis - t.ij_dis  for t in triplets]
println(mean(diss))
println(std(diss))
# clamp!(diss,-1,1)
# diss = rand(100)
# println(diss)
h = fit(Histogram, diss, nbins=100)
using Plots
# plotly()
histogram(diss)
