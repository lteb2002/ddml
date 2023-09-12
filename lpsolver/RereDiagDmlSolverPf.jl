

#
module RereDiagDmlSolverPf
using LinearAlgebra
using Optim, LineSearches
export solveDmlLp
using Statistics

# 本计算中带有DML松驰变量（DML松弛变量+等式松驰变量）

  # Solve the L2 regularization LP problem for Diag-DML based on Penalty Function
  function solveDmlLp(c,A,b, regType::String="l2", regWeight::Number=0, a::Number = 0.5)
    c = Float32.(c)
    A = Float32.(A)
    b = Float32.(b)
    n_inst = length(b)
    n_feature=length(c)-2*n_inst
    if ndims(A) == 1
      # Julia的reshape为先列再行的格式，因此需要转置
      A = transpose(reshape(A,(length(c),length(b))))
      println("1D array of A is converted to 2D array.")
    end
    # println("c:",c)
    # println("A:",A)
    # println("b:",b)
    x0=ones(length(c))*1.5   #x初始值
    alpha=1.0 #tho增长系数
    eps = Float32.(1.0e-100)
    param_c = Float32.(1.0E-100) #L1拟合函数中的参数c
    #G=ones(length(c))

    # function ctr(x)
    #   #println(x)
    #   ct= -  sum(log.(indicator(x))) +  sum((A*x-b)^2)
    #   return ct
    # end

    # 计算罚函数
    function bf(x)
      # y1=[e<0 ? e^2 : 0.0 for e in x]
      y1 = zeros(length(x))
      indices = x .< 0
      y1[indices] = x[indices] .^ 2
      y2 = (A*x-b).^2
      return sum(y1)+sum(y2)
    end


    function bf_gra(x)
      y1 = zeros(length(x))
      indices = x .< 0
      y1[indices] = x[indices] .* 2.0
      # y1 = [e < 0 ? 2.0 * e : 0 for e in x]
      temps = A*x-b
      # temps2=[t<0 ? t : 0.0 for t in temps]
      y2 = 2.0 * A' * temps
      return y1+y2
    end

    function pure_objective(x)
      # y = c' * x + alpha * bf(x) + regWeight * compute_reg_terms(x,regType)[1]
      y = c[1:n_feature]' * x[1:n_feature]  + regWeight * compute_reg_terms(x[1:n_feature],regType)[1]
      return y
    end

    function sign_for_l1(params)
      # 方法1
      # y = sum(abs.(params))
      # 方法2
      y = sum((params .^2 .+ eps).^0.5)
      # 方法3
      # y = sum([abs(p) >= param_c ? abs(p) : p^2/(2*param_c)+param_c/2.0 for p in params])
      # y = zeros(length(params))
      # indices = abs.(params) .< param_c
      # y[indices] = params[indices] .^2 ./(2*param_c) .+ param_c/2.0
      # ind1 = params .>= param_c
      # y[ind1] .= params[ind1]
      # ind2=params .<= param_c
      # y[ind2] .= -params[ind2]
      # y = sum(y)
      return y
    end

    # 计算L1正则项的梯度
    function sign_for_l1_gradient(params)
      gra = zeros(length(params))
      # 方法1
      # gra[params .> 0] .= 1
      # gra[params .< 0] .= -1
      # gra[params .== 0] .= 0
      # 方法2
      gra = (params .^2 .+ eps).^-0.5 .* 0.5 .* 2 .* params
      # 方法3
      # gra = [p/param_c for p in params]
      # indices = abs.(params) .< param_c
      # gra[indices] = gra[indices] ./ param_c
      # gra[params .>= param_c] .= 1
      # gra[params .<= param_c] .= -1
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

    p_slack = 1.0 # 似乎在构建系数时已加了惩罚mu
    #原函数
    function f(x)
      reg = regWeight * compute_reg_terms(x,regType)[1]
      # println(regWeight,reg)
      f(x)=c' * x + alpha * bf(x) + reg
    end
    # f(x)=c' * x + alpha * bf(x) + (regWeight * compute_reg_terms(x,regType)[1]) + p_slack*sum(abs.(x[n_feature+1:n_feature+n_inst]))
    #f(x)=c' * x - alpha * sum(log.(indicator(x))) - alpha * sum(log.(indicator(-A*x+b)))
    #f(x) = c' * x + alpha * sum(1.0 ./ x) - alpha * sum(1.0 ./ (A*x-b))
    #梯度函数
    function grad(xx) 
      # reg = regWeight * compute_reg_terms(xx,regType)[1]
      # println(regWeight,",",reg,",a:",a)
      return c + alpha * bf_gra(xx) + regWeight * compute_reg_terms(xx,regType)[2]
    end
    function g!(G, xx)
      G .= grad(xx)
      # G[n_feature+1:n_feature+n_inst] .+= p_slack*[e>0 ? 1 : -1 for e in xx[n_feature+1:n_feature+n_inst]]
      #G .= c - alpha * (1 ./ xx .^2) + alpha * A * (1 ./ (A*xx-b).^2 )
    end

    println("f:",f(x0))
    #println("g:",g!(G,x0))
    maxStep=1000
    error1=1.0
    error2=1.0
    currentStep = 0
    objs = zeros(0)
    threshold = 1.0E-16
    while error1>threshold && currentStep < maxStep
      currentStep += 1
      res=optimize(f, g!,x0, ConjugateGradient())
      x0 = Optim.minimizer(res)
      #println("result = ", x0 )
      error1 = bf(x0)
      pobj=pure_objective(x0)
      append!( objs, pobj)
      if error1>threshold
        alpha *= 3
      else
        if length(objs) >= 3
          if std(objs[end-3:end]) < threshold
            break
          end
        end
      end
      error2 = mean(abs.(grad(x0)))
      println("Pf Step:",currentStep,",Objective value: ",pobj,", bf:",error1,", residual:",error1*alpha)
      #error=sum(abs.(ctr(x0)))
    end
    #println("result = ", x0 )
    # println("error:",error)
    x0 = round.(x0;digits=6)
    return x0
  end


end
