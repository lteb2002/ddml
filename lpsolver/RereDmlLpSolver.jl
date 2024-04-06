

#This module works as a linear programming based solver for DDML problem
module RereDmlLpSolver
using JuMP,Clp
using HiGHS
export solveDmlLp

# Just for testing purpose, Diag-DML and its ℓ_1 regularization solve DDML problems using a Scala library called "SCPSolver.jar".
#solve linear programming
function solveDmlLp(
  c::Array{Float32},
  A::Array{Float32,2},
  b::Array{Float32},
  regWeight::Number = 0
)
  # @suppress
  # model = Model(HiGHS.Optimizer)
  model = JuMP.Model(Clp.Optimizer)
  set_optimizer_attribute(model, "LogLevel", 0)
  # set_optimizer_attribute(model, "SolveType", 0)
  #println(length(c))
  @variable(model, x[i = 1:length(c)])
  @constraint(model, con, A * x .>= b)
  @constraint(model, con0, x .>= 0)
  @objective(model, Min, c' * x + regWeight * sum(x))
  status = optimize!(model)
  result = JuMP.value.(x)
  obj = JuMP.objective_value(model)
  #println("x = ", result)
  #println("Objective value: ", obj)
  return result
end
end
