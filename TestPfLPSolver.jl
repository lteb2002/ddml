

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
using RereDmlLpSolverPf
c1=Vector{Float32}([-3, -1, -2, 0, 0, 0])
b1= Vector{Float32}([30, 24, 36])
A1= Matrix{Float32}([1.0  1.0  3.0 1 0 0;
                     2.0  2.0  5.0 0 1 0;
                     4.0  1.0  2.0 0 0 1])
@time x=RereDmlLpSolverPf.solveDmlLp(c1,A1,b1,"l2",0)
println(x)
println(c1'*x)
println(A1*x-b1)


c1=Vector([-2, -4, -1,-1, 5,5,5,5,0, 0, 0,0])
b1= Vector([8, 6, 6,9])
A1= Matrix([1.0  3.0  0.0  4.0  1  0  0  0  -1  0  0  0;
            2.0  1.0  0.0  0.0  0  1  0  0   0 -1  0  0;
            0.0  1.0  1.0  1.0  0  0  1  0   0  0 -1  0;
            1.0  1.0  1.0  0.0  0  0  0  1   0  0  0 -1])

@time x=RereDmlLpSolverPf.solveDmlLp(c1,A1,b1,"l2",10)
println("Pf solution",x[1:10])
