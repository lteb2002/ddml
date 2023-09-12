

#set the path where the modules are located
push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
push!(LOAD_PATH, "./ju4ja")

#A examle function ready to be called from java using Ju4ja
function greetings(name::String)
    str="Hi, $name"
    println(str)
    return str
end

# g=greetings("test")
# println(g)

using Ju4ja
# using ArrayFire
#include("RereDmlLpSolverY2.jl")
using RereDmlLpSolverPf
c1=[-3, -1, -2, 0, 0, 0]
b1= [30, 24, 36]
A1= [1.0  1.0  3.0 1 0 0; 2.0  2.0  5.0 0 1 0; 4.0  1.0  2.0 0 0 1]

@time x0=solveDmlLp(c1,A1,b1,"none",0.0)
println(x0)

#using Ju4ja and the example solver for linear programming
# using Ju4ja
# #include("RereDmlLpSolverY2.jl")
# using RereDmlLpSolverY2
# c1=Vector([-3, -1, -2, 0, 0, 0])
# b1= Vector([30, 24, 36])
# A1= Matrix([1.0  1.0  3.0 1 0 0; 2.0  2.0  5.0 0 1 0; 4.0  1.0  2.0 0 0 1])
# @time RereDmlLpSolverY2.solveDmlLp(c1,A1,b1,"none",0.0)


#start the Ju4ja server as a new coroutine
@async begin
    startServer(port=6996)
end

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
push!(LOAD_PATH, "./ju4ja")
using Ju4ja
using RereDmlLpSolverPf
json="{\"id\":\"5bda9839-d6c2-4830-b11a-10c681f8e6f9\",\"operation\":\"FUNCTION\",\"script\":null,\"modn\":\"RereDmlLpSolverPf\",\"func\":\"solveDmlLp\",\"args\":[[[-3.0,-1.0,-2.0,0.0,0.0,0.0],[1.0,1.0,3.0,1.0,0.0,0.0,2.0,2.0,5.0,0.0,1.0,0.0,4.0,1.0,2.0,0.0,0.0,1.0],[30.0,24.0,36.0],\"L2\",0.0,3]],\"resultType\":null}"
result=Ju4ja.parseAndExecute(json)
println(result)
