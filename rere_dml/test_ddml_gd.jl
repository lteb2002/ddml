


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

#本文件用于测试DDML（对角化度量学习）的A传统求解器，如乘子法、罚函数法、和线性规划方法（仅能用于原问题或者L1）
#This file is used to test the performance of the traditional solvers for DDML(Diagonal Distance Metric Learning)

path="G:\\dataset\\dml_feature_selection_data\\"
f = "credit_score2"


fn=path*f*".csv"

println("Loading data from $fn--------------------------")
csv = CSV.read(fn,DataFrame)
# data = convert(Array{Float16,2}, csv[:,1:end-1])
data = Array{Float16,2}(csv[:,1:end-1])
dt = fit(UnitRangeTransform, data, dims=1)
# dt = fit(ZScoreTransform, data, dims=1)
data = StatsBase.transform(dt, data)
# println(data)
labels = "label_".*string.(csv[:,end])
# println(labels)

reg = "l2"
reg = "elastic"
reg = "none"
# reg = "l1"
distance_type = "huber2"
regWeight = 10^2.5
triplets = TripletModule.build_triplets(data, labels)
@time x=DiagDml.solve_diag_dml(triplets,regWeight,0.8,distance_type)
x = round.(x;digits=12)
println("Solutions:",x)
new_data = data * Diagonal(x)
new_data = round.(new_data;digits=12)
# println(new_data)
csv = hcat(new_data,labels)
# println(csv)
output = path*f*"_"*reg*"_"*distance_type*".csv"
table = Tables.table(csv)
CSV.write(output,table)

using Plots
plotly()
bar(x)
