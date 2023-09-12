

#
module RereDmlLpSolverBox
using LinearAlgebra
using Optim
export solveDmlLp

  #
  function solveDmlLp(c::Vector,A::Matrix,b::Vector;ifReg::Bool=false)
    v=ones(length(b))  #拉格朗日项系数
    tho=20 #增广拉格朗日项系数（二次）
    x0=zeros(length(c)) .+ 0.5 #x初始值
    alpha=3 #tho增长系数

    lower = zeros(length(c))
    upper = zeros(length(c))
    upper .= Inf

    #G=ones(length(c))

    #原函数
    f(x)=c' * x - v' * (A * x - b) + (tho / 2.0) .* ((A * x - b)' * (A * x - b)) + ( ifReg ? x' * x : 0)
    #梯度函数
    function g!(G, xx)
      G .= c - A' * v + tho .* A' * (A * xx - b) +  ( ifReg ? 2 .* xx : zeros(length(xx)))
    end
    function cntr(xx)
      A * xx  - b
    end
    println("f:",f(x0))
    #println("g:",g!(G,x0))
    println("ct:",cntr(x0))
    maxStep=1000
    error=1.0
    currentSetp = 0
    while error>1.0E-4 && currentSetp < maxStep
      currentSetp += 1
      res=optimize(f, g!,lower, upper, x0, Fminbox(ConjugateGradient()))
      x0 = Optim.minimizer(res)
      #println("result = ",x0 )
      #println("Objective value: ",Optim.minimum(res))
      constrainError = cntr(x0)
      tho = alpha * tho #更新增广拉格朗日项系数
      v = v - tho * constrainError #更新拉格朗日项系数向量
      error = norm(constrainError)
      println("error:",error)
    end
    println(x0)
    return x0
  end


end
