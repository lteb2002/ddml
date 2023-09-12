

#
module RereDiagDmlSolverPf2
using LinearAlgebra
using Optim, LineSearches
export solveDmlLp
using Statistics

# 本计算中没有带有松驰变量（不等式松驰变量）

  # Solve the L2 regularization LP problem for Diag-DML based on Penalty Function
  function solveDmlLp(c,A,b, regType::String="l2", regWeight::Number=0, a::Number = 0.5)
    c = Float32.(c)
    A = Float32.(A)
    b = Float32.(b)
    n_inst = length(b)
    # n_feature=length(c)-2*n_inst
    if ndims(A) == 1
      # Julia的reshape为先列再行的格式，因此需要转置
      A = transpose(reshape(A,(length(c),length(b))))
      println("1D array of A is converted to 2D array.")
    end
    # println("c:",c)
    # println("A:",A)
    # println("b:",b)
    x0=ones(length(c))*1.5   #x初始值
    beta1=1.0 #tho增长系数
    beta2=1.0 #tho增长系数
    eps = 1.0e-100
    param_c = 1.0E-10
    residual = 0
    #G=ones(length(c))

    # function ctr(x)
    #   #println(x)
    #   ct= -  sum(log.(indicator(x))) +  sum((A*x-b)^2)
    #   return ct
    # end

    #变量非负约束
    function bf1(x)
      y1 = zeros(length(x))
      indices = x .< 0
      y1[indices] = x[indices] .^ 2
      return sum(y1)
    end

    #数据点的异类间隔大于同类间隔约束
    function bf2(x)
      temps = A*x-b
      indices2 = temps .< 0
      y2 = zeros(length(temps))
      y2[indices2] .= temps[indices2] .^ 2
      return sum(y2)
    end

    # 计算罚函数
    function bf(x)
      s1 = beta1 * bf1(x)
      s2 = beta2 * bf2(x)
      # y2=[t<0 ? t^2 : 0.0 for t in temps]
      return s1+s2
    end

    # 计算罚函数的梯度
    function bf_gra(x)
      # y1 = [e < 0 ? 2.0 * e : 0 for e in x]
      y1 = zeros(length(x))
      indices = x .< 0
      y1[indices] = x[indices] .* 2.0
      temps = A*x-b
      # println(length(temps),length(x))
      indices2 = temps .< 0
      # y2 = zeros(length(temps))
      # y2[indices2] .= temps[indices2]
      valid_f = temps[indices2]
      # temps2=[t<0 ? t : 0.0 for t in temps]
      y2 = 2.0 * A[indices2,:]' * valid_f
      return (beta1*y1+beta2*y2)
    end

    function pure_objective(x)
      y = c' * x  + regWeight * compute_reg_terms(x,regType)[1]
      return y
    end


    function sign_for_l1(params)
      # 方法1
      # y = sum(abs.(params))
      # 方法2
      # y = sum((params .^2 .+ eps).^0.5)
      # 方法3
      # y = sum([abs(p) >= param_c ? abs(p) : p^2/(2*param_c)+param_c/2.0 for p in params])
      y = zeros(length(params))
      indices = abs.(params) .< param_c
      y[indices] = params[indices] .^2 ./(2*param_c) .+ param_c/2.0
      ind1 = params .>= param_c
      y[ind1] .= params[ind1]
      ind2=params .<= param_c
      y[ind2] .= -params[ind2]
      return sum(y)
    end

    # 计算L1正则项的梯度
    function sign_for_l1_gradient(params)
      gra = zeros(length(params))
      # 方法1
      # gra[params .> 0] .= 1
      # gra[params .< 0] .= -1
      # gra[params .== 0] .= 0
      # 方法2
      # gra = (params .^2 .+ eps).^-0.5 .* 0.5 .* 2 .* params
      # 方法3
      indices = abs.(params) .< param_c
      gra[indices] = gra[indices] ./ param_c
      gra[params .>= param_c] .= 1
      gra[params .<= param_c] .= -1
      return gra
    end

    # 计算正则项
    function compute_reg_terms(params,reg_type="elastic")
      # println(reg_type)
      l1,l2=a,(1-a)
      obj_p,grad_p=0,zeros(length(params))
      if lowercase(reg_type) == "l2"
        obj_p = sum(params .^ 2)
        grad_p = 2.0 * params
      elseif lowercase(reg_type) == "elastic"
        obj_p = l2*sum(params .^ 2) + l1*sign_for_l1(params)
        grad_p = l2*2.0 * params + l1*sign_for_l1_gradient(params)
      elseif lowercase(reg_type) == "l1"
        obj_p = sign_for_l1(params)
        grad_p = sign_for_l1_gradient(params)
      end
      return (obj_p,grad_p)
    end

    #原函数
    f(x)=c' * x + bf(x) + (regWeight * compute_reg_terms(x,regType)[1])
    #f(x)=c' * x - alpha * sum(log.(indicator(x))) - alpha * sum(log.(indicator(-A*x+b)))
    #f(x) = c' * x + alpha * sum(1.0 ./ x) - alpha * sum(1.0 ./ (A*x-b))
    #梯度函数
    function grad(xx) 
      return c + bf_gra(xx) + regWeight * compute_reg_terms(xx,regType)[2]
    end
    function g!(G, xx)
      G .= grad(xx)
      # G[n_feature+1:n_feature+n_inst] .+= p_slack*[e>0 ? 1 : -1 for e in xx[n_feature+1:n_feature+n_inst]]
      #G .= c - alpha * (1 ./ xx .^2) + alpha * A * (1 ./ (A*xx-b).^2 )
    end

    println("f:",f(x0))
    #println("g:",g!(G,x0))
    maxStep=200
    error1=1.0
    error2=1.0
    error3=1.0
    currentStep = 0
    # objs = zeros(0)
    bf2s = zeros(0)
    threshold = 1.0E-4
    threshold2 = 1.0E-64
    residual = 100
    while residual>threshold && currentStep < maxStep
      currentStep += 1
      res=optimize(f, g!,x0, ConjugateGradient())
      x0 = Optim.minimizer(res)
      #println("result = ", x0 )
      error1 = bf1(x0)
      error2 = bf2(x0)
      pobj=pure_objective(x0)
      # append!( objs, pobj)
      append!(bf2s,error2)
      thor = 2
      beta1 *= thor
      beta2 *= thor
      if length(bf2s) >= 5
        error3 = std(bf2s[end-5+1:end]) 
      end
      residual = beta1*error1+beta2*error2
      println("Pf Step:",currentStep,",Objective value: ",pobj,", bf1:",error1,", bf2:",error2,", residual:",residual)
      #error=sum(abs.(ctr(x0)))
    end
    #println("result = ", x0 )
    # println("error:",error)
    return x0
  end

end
