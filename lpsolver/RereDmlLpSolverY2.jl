

#
module RereDmlLpSolverY2
using LinearAlgebra
using Optim
export solveDmlLp

  #
  function solveDmlLp(c,A,b, regType::String="l2", regWeight::Number=0,mainVarNum::Number=0)
    c = Float32.(c)
    A = Float32.(A)
    b = Float32.(b)
    println("A:",A)
    regType = lowercase(regType)
    v=ones(size(A)[1])  #拉格朗日项系数
    tho=20 #增广拉格朗日项系数（二次）
    x0=zeros(length(c)) .+ 0.5 #x初始值
    alpha=3 #tho增长系数
    println(A * (x0 .* x0) - b)
    #原函数
    f(x)=c' * (x .* x) - v' * (A * (x .* x) - b) + (tho / 2.0) .* ((A * (x .* x) - b)' * (A * (x .* x) - b)) + (regType=="l2" ? regWeight*(x' * x)^2 : 0)
    #梯度函数
    function g!(G, xx)
      G .= 2.0 .* (c .* xx) - (2.0 .* A' * v) .* xx + (2.0 * tho .* A' * (A * (xx .* xx) - b)) .* xx +( regType == "l2" ? 2*regWeight*(xx' * xx) .* xx : zeros(length(xx)) )
    end
    function cntr(xx)
      A * (xx .* xx) - b
    end
    println("f:",f(x0))
    println("ct:",cntr(x0))
    maxStep=100
    error=1.0
    currentSetp = 0
    while error>1.0E-4 && currentSetp < maxStep
      currentSetp += 1
      res = optimize(f, g!, x0, LBFGS(),Optim.Options(g_tol = 1e-3,
                             iterations = 100,
                             store_trace = true,
                             show_trace = false))
      x0 = Optim.minimizer(res)
      #println("result = ",x0 )
      #println("Objective value: ",Optim.minimum(res))
      constrainError = cntr(x0)
      tho = alpha * tho #更新增广拉格朗日项系数
      v = v - tho * constrainError #更新拉格朗日项系数向量
      error = norm(constrainError)
      #println("error:",error)
      #println("tho:",tho)
    end
    x0= x0 .* x0
    println(x0)
    return x0
  end


end
