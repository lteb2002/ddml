
push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")

using LinearAlgebra
using RereDmlLpSolver

c1=Vector{Float32}([-3, -1, -2, 0, 0, 0])
b1= Vector{Float32}([30, 24, 36])
A1= Matrix{Float32}([1.0  1.0  3.0 1 0 0;
                     2.0  2.0  5.0 0 1 0;
                     4.0  1.0  2.0 0 0 1])
@time x=RereDmlLpSolver.solveDmlLp(c1,A1,b1,0)
println(x)
println(c1'*x)
println(A1*x-b1)
