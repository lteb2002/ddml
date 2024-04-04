

#
module RereDiagDmlSolverLangMul
using LinearAlgebra
using Optim, LineSearches
export solveDmlLp
using Statistics

# 本模块使用乘子法测试求解DDML问题
# 本计算中没有带有解决不等式约束的剩余变量（非不等式松驰变量）

  # Solve the L2 regularization LP problem for Diag-DML based on Penalty Function
  function solveDmlLp(c,A,b, regType::String="l2", regWeight::Number=0, a::Number = 0.5)
    c = Float16.(c)
    A = Float16.(A)
    b = Float16.(b)
    n_inst = length(b)
    m_feature = length(c)-length(b)
    # n_feature=length(c)-2*n_inst
    if ndims(A) == 1
      # Julia的reshape为先列再行的格式，因此需要转置
      A = transpose(reshape(A,(length(c),length(b))))
      println("1D array of A is converted to 2D array.")
    end
    # println("c:",c)
    # println("A:",A)
    # println("b:",b)
    x0=ones(length(c))  #x初始值
    tho = 1.0E6 #tho 固定数值的惩罚系数
    beta_lang = ones(2*n_inst+m_feature)
    # beta_lang = ones(2*n_inst+m_feature)
    eps = 1.0e-100
    param_c = 1.0E-10 # 1范数拟合函数参数
    residual = 0
    #G=ones(length(c))

    # function ctr(x)
    #   #println(x)
    #   ct= -  sum(log.(indicator(x))) +  sum((A*x-b)^2)
    #   return ct
    # end

    function cal_constraint_item_value(x)
      y = zeros(2*n_inst+m_feature)
      y[1:n_inst] = A*x-b
      y[n_inst+1:n_inst+m_feature] = x[1:m_feature]
      y[n_inst+m_feature+1,end]=x[m_feature+1,end]
      y2 = zeros(2*n_inst+m_feature)
      indice = y .< 0
      y2[indice] = y[indice]
      return y2
    end

    #数据点的异类间隔大于同类间隔约束及变量非负约束
    function bf(x)
      # 个数为约束不等式的个数，n个约束, m个主变量, n个松弛变量
      y = zeros(2*n_inst+m_feature)
      #本部分代表DML最大小间隙的约束
      y[1:n_inst] = beta_lang[1:n_inst]-tho*(A*x-b) # 参见P408，最优化理论与方法（第2版），陈宝林
      # 相较于上书411中的公式，本问题中多了x>0和\xi（希腊字符）>0的约束
      #本部分代表x>0
      y[n_inst+1:n_inst+m_feature] = beta_lang[n_inst+1:n_inst+m_feature]-tho*x[1:m_feature]
      #本部分代表DML松驰变量>0
      y[n_inst+m_feature+1,end]=beta_lang[n_inst+m_feature+1,end]-tho*x[m_feature+1,end]

      indices = y .> 0
      # println(length(y))
      # println(length(indices))
      y2 = zeros(2*n_inst+m_feature)-beta_lang .^ 2
      # println(length(y2))
      xx = y[indices] .^ 2 
      y2[indices] += xx
      return sum(y2)/(tho*2)
    end

    # 计算罚函数的梯度
    function bf_gra(x)
      y1 = zeros(length(x))
      indice1 = (beta_lang[n_inst+1:end]-tho*x) .> 0
      y1[indice1] .= 2.0*(beta_lang[n_inst+1:end][indice1] - tho * x[indice1])*(-tho)

      y2 = zeros(length(x))
      indice2 = (beta_lang[1:n_inst]-tho*(A*x-b)) .> 0
      valid_f = beta_lang[1:n_inst][indice2] - tho*(A*x-b)[indice2]
      y2 = 2.0 * A[indice2,:]' * valid_f * (-tho)

      return (y1+y2)./(tho*2)
    end

    # 
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


    function lipschitz_k(x1,x2)
      k = (f(x1)-f(x2))/(norm(x1-x2))
      k = abs(k)
      return k
    end

    function test_lipschitz()
      x1=ones(length(c))
      x2=ones(length(c))*1.0E-12
      return lipschitz_k(x1,x2)
    end

    #梯度函数
    function grad(xx) 
      return c + bf_gra(xx) + regWeight * compute_reg_terms(xx,regType)[2]
    end
    function g!(G, xx)
      G .= grad(xx)
    end

    # println("f:",f(x0))
    #println("g:",g!(G,x0))
    maxStep=200
    error1=1.0
    currentStep = 0
    # objs = zeros(0)
    k0 = test_lipschitz()
    gx0 = norm(grad(x0))
    println("Current Lipschitz K:",k0,", gradient norm:",gx0)
    threshold = 1.0E-4
    residual = 100
    h0 = norm(cal_constraint_item_value(x0))

    # Default nonlinear procenditioner for `OACCEL`
    nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
    linesearch=LineSearches.Static())
    nlprecon2 = LBFGS()
    # Default size of subspace that OACCEL accelerates over is `wmax = 10`
    oacc10 = OACCEL(nlprecon=nlprecon2, wmax=10)

    while residual>threshold && currentStep < maxStep
      currentStep += 1
      #ConjugateGradient  LBFGS
      res=optimize(f, g!,x0, oacc10)
      x0 = Optim.minimizer(res)
      beta_lang -= tho * cal_constraint_item_value(x0)
      #println("result = ", x0 )
      h1 = norm(cal_constraint_item_value(x0))
      rate = h1/h0
      if  rate >= 0.1
        tho = tho * 10
      end
      # println(rate,":",tho)
      h0=h1
      error1 = h1
      pobj=pure_objective(x0)
      # append!( objs, pobj)
      residual = error1
      println("Pf Step:",currentStep,",Objective value: ",pobj,", bf1:",error1,",  residual:",residual)
      #error=sum(abs.(ctr(x0)))
    end
    println("Total steps:",currentStep)
    #println("result = ", x0 )
    # println("error:",error)
    return x0
  end

end
