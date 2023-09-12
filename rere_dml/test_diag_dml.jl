


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

path="G:\\dataset\\dml_feature_selection_data\\"
f = "credit_score2"

path="G:\\bayes\\data\\"
f = "credit_sample"

fn=path*f*".csv"

println("Loading data from $fn--------------------------")
csv = CSV.read(fn,DataFrame)
# data = convert(Array{Float32,2}, csv[:,1:end-1])
data = Array{Float32,2}(csv[:,1:end-1])
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
regWeight = 1.0
triplets = TripletModule.build_triplets(data, labels)
@time x=DiagDml.solve_diag_dml(triplets,regWeight,0.0,distance_type)
x = round.(x;digits=4)
println("Solutions:",x)
new_data = data * Diagonal(x)
new_data = round.(new_data;digits=4)
# println(new_data)
csv = hcat(new_data,labels)
# println(csv)
output = path*f*"_"*reg*"_"*distance_type*".csv"
table = Tables.table(csv)
CSV.write(output,table)

using Plots
plotly()
bar(x)
