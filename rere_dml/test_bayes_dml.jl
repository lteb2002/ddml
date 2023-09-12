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
using BenchmarkTools
# Import Turing and Distributions.
using Turing, Distributions
using LinearAlgebra

# Import RDatasets.
using RDatasets

# Import MCMCChains, Plots, and StatPlots for visualizations and diagnostics.
using MCMCChains, StatsPlots

# Functionality for splitting and normalizing the data.
using MLDataUtils: shuffleobs, splitobs, rescale!

# Functionality for evaluating the model predictions.
using Distances

# Set a seed for reproducibility.
using Random
Random.seed!(0)

# Hide the progress prompt while sampling.
Turing.setprogress!(true);

path="G:\\bayes\\data\\"
f = "credit_sample"

fn=path*f*".csv"

println("Loading data from $fn--------------------------")
csv = CSV.read(fn,DataFrame)
# data = convert(Array{Float32,2}, csv[:,1:end-1])
data = Array{Float32,2}(csv[:,1:end-1])
dt = fit(ZScoreTransform, data, dims=1)
data = StatsBase.transform(dt, data)

# println(typeof(data))
# println(data)
labels = "label_".*string.(csv[:,end])
# println(labels)
triplets = TripletModule.build_triplets(data, labels)
xjs = hcat([t.xj for t in triplets]...)
xis = hcat([t.xi for t in triplets]...)
jis = convert(Array{Float64,2},(xjs - xis))
xks = hcat([t.xk for t in triplets]...)
jks = convert(Array{Float64,2},(xjs - xks))
# println(typeof(jis))
# println(typeof(jks))
data = convert(Array{Float64,2},data)

nfeatures = size(data, 2)
nsamples = size(jis,2)
println(nsamples)
# addprocs(8)
function cc(ws,jxs)
    trans = @views ws .* jxs
    # println(size(trans))
    l2s = norm.(eachcol(trans))
    # l2s = map(trans,norm)
    # println(size(l2s))
    return l2s
end
# Bayesian linear regression.
@model function bayes_dml(jis2,jks2) where {T}
    # Set the priors on our coefficients.
    # 压缩ws的值（方差），可以防止过拟合
    ws ~ MvNormal(zeros(nfeatures), sqrt(5)*I)
    # tau ~ truncated(Normal(1, 100), 0, Inf)
    # eps ~ MvNormal(ones(nsamples)*2, 10)
    sims = cc(ws,jis2)
    difs = cc(ws,jks2)
    gap = difs - sims
    # y_hat = gap - eps
    # println(gap)
    t = 1
    y = rand(Normal(2, t),nsamples) 
    # y = ones(nsamples) 
    # Set variance prior.
    σ₂ ~ truncated(Normal(t, 10), 0, Inf)
    y ~ MvNormal( gap, σ₂*I)
end

# yy = rand(Normal(1,0.0001),nsamples)
# println(yy)
model = bayes_dml(jis,jks)
chain = sample(model, HMC(0.1, 5), 3_000);
describe(chain)
# pyplot()
plot(chain)
wws = group(chain, "ws").value.data[:,:,1]
# print(wws)
ms = mean.(eachcol(wws))
ms = round.(abs.(ms);digits=4)
print(ms)
new_data = data * Diagonal(ms)
new_data = round.(new_data;digits=4)
# println(new_data)
csv = hcat(new_data,labels)
# println(csv)
output = path*f*"_bayes.csv"
table = Tables.table(csv)
CSV.write(output,table)
# histogram(chain["ws[2]"])
# savefig(path*"gdemo-plot.png")
