

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
#include("RereDmlLpSolverY2.jl")
using RereDmlLpSolverY2
c1=Vector([-3, -1, -2, 0, 0, 0])
b1= Vector([30, 24, 36])
A1= Matrix([1.0  1.0  3.0 1 0 0; 2.0  2.0  5.0 0 1 0; 4.0  1.0  2.0 0 0 1])
@time RereDmlLpSolverY2.solveDmlLp(c1,A1,b1,"none",0.0)


push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
#include("RereDmlLpSolverY2.jl")
using RereDmlLpSolverBox
c1=Vector([-3, -1, -2, 0, 0, 0])
b1= Vector([30, 24, 36])
A1= Matrix([1.0  1.0  3.0 1 0 0; 2.0  2.0  5.0 0 1 0; 4.0  1.0  2.0 0 0 1])
@time RereDmlLpSolverBox.solveDmlLp(c1,A1,b1)



push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
using RereDmlLpSolverBf
c1=Vector([-3, -1, -2, 0, 0, 0])
b1=  Vector([30, 24, 36])
A1= Matrix([1.0  1.0  3.0 1 0 0; 2.0  2.0  5.0 0 1 0; 4.0  1.0  2.0 0 0 1])
@time x=RereDmlLpSolverBf.solveDmlLp(c1,A1,b1,"l2",1.0)
println(x)
println(c1'*x)
println(A1*x-b1)



#RereDmlLpSolverY2.solveDmlLp(c,A,b)

y= map(x->begin
if x == 0 return 0
elseif iseven(x) return 2
elseif isodd(x) return 1
end
end,collect(-3:3)
)
@elapsed println(y)


using LinearAlgebra
using Optim
#原函数
f(x) = x' * x - [2.0,2,2]' * x + 1.0
#梯度函数
function g!(G, xx)
  G .= 2xx -[2.0,2,2]
end
res=optimize(f,  [1.0,1,1], LBFGS();autodiff = :forward)
x0 = Optim.minimizer(res)
println(x0)



using LinearAlgebra
using Optim
#原函数
f(x)= x'x - [2]' * x + 1.0
#梯度函数
function g!(G, xx)
  G .= 2 .* xx .- [2]
  return G
end
x0=[1.0]
G=[2.0]
println("g:",g!(G,x0))
println("G:",G)
res=optimize(f,g!,[1.0], LBFGS())
x0 = Optim.minimizer(res)
println(x0)
