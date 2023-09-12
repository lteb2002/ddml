

function greetings()
println("hello world")
end


greetings()

using DataArrays
a = DataArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])

A=rand(3,3)
print(A)

#计算导数
using Calculus
f(x) = sin(x)
y=f(1.0) - cos(1.0)
println(y)

fx,fy=differentiate("cos(x) + sin(y) + exp(-x) * cos(y)", [:x, :y])

using TimeSeries
dates = collect(Date(2017,8,1):Date(2017,8,5))
println(typeof(dates))
sample_time = TimeArray(dates, rand(length(dates)))


using JuMP
using GLPK
m = JuMP.Model(with_optimizer(GLPK.Optimizer))
@variable(m, 0 <= a <= 2 )
@variable(m, 0 <= b <= 10 )
@objective(m, Max, 5a + 3*b )
@constraint(m, 1a + 5b <= 3.0 )
print(m)
status = optimize!(m)
println("Objective value: ", JuMP.objective_value(m))
println("a = ", JuMP.value(a))
println("b = ", JuMP.value(b))



using PyPlot
x = range(0,stop=2*pi,length=1000)
y = sin.(3*x + 4*cos.(2*x))
p=plot(x, y, color="red", linewidth=2.0, linestyle="--")




push!(LOAD_PATH, "./")
using Ju4ja
using RereDmlLpSolver
json="{\"id\":\"b08ef401-0912-4ce1-8d9b-a0306c7acdd7\",\"operation\":\"FUNCTION\",\"script\":null,\"modn\":\"RereDmlLpSolver\",\"func\":\"solveDmlLp\",\"args\":[[-3.0,-1.0,-2.0],[[-1.0,-1.0,-3.0],[-2.0,-2.0,-5.0],[-4.0,-1.0,-2.0]],[-30.0,-24.0,-36.0]],\"resultType\":null}"#println(json)
result=Ju4ja.parseAndExecute(json)
println(result)

push!(LOAD_PATH, "./")
using RereDmlLpSolver
test = ([-3.0, -1.0, -2.0], [-1.0 -1.0 -3.0; -2.0 -2.0 -5.0; -4.0 -1.0 -2.0], [-30.0, -24.0, -36.0])
@time for i in 1:10000
    RereDmlLpSolver.solveDmlLp(test...)
end


nargs = ([-3.0, -1.0, -2.0], [-1.0 -1.0 -3.0; -2.0 -2.0 -5.0; -4.0 -1.0 -2.0], [-30.0, -24.0, -36.0])
result=getfield(Main, Symbol("solveDmlLp"))(nargs...)


args=(Any[-3.0, -1.0, -2.0], Any[Any[1.0, 1.0, 3.0], Any[2.0, 2.0, 5.0], Any[4.0, 1.0, 2.0]], Any[30.0, 24.0, 36.0])
include("Ju4jaParser.jl")
ls=parseAllParams(args)
println(ls)



x=[1.0, 1.0, 3.0]
println(typeof(x))
println(isa(x,Array{Float64,1}))
println(all(y isa Number for y in x))
println(vcat(x...))


t=0.0
println(t isa Number)

y=Any[Any[1.0, 1.0, 3.0], Any[2.0, 2.0, 5.0], Any[4.0, 1.0, 2.0]]
println(typeof(y))
println(all(t isa Array for t in y))
println(hcat(y...)')
println(size(y))



a=[-3.0, -1.0, -2.0]
b=[ x+1 for x in a]
println(b)


tt=Any[1.0, 1.0, 3.0]
println([x for x in tt])

using Distributed
for pid in workers()
    println(pid)
end

using ArrayFire, LinearAlgebra
a = AFArray(rand(3000,3000))
@time F=svd(a)

b=rand(3000,3000)
@time F=svd(b)


svs=F[2]
fw= svs[1]/sum(svs)


x1= AFArray([1 2 3])
x2 = AFArray([4 5 6])
x3= vcat(x1,x2)
x4=x3[1:1,:]
x= AFArray([1 2 3])
yy=Matrix(x)
println(yy)

using LinearAlgebra
a1=Array(1:5)
a2=
