
module RereDiagDmlADMM

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./rere_dml")
push!(LOAD_PATH, "./lpsolver")

using DiagDml
using RereDmlLpSolver
using LinearAlgebra
using Statistics
using Optim, LineSearches
using Random
# using Distributed

export admmIterate

#本模块为Diagonal DML优化问题的主要求解器，分块方法为按样本分裂，可以用于并行计算
# This module is the main solver of the DDML optimization, it can split the problem with many blocks of samples, and it can be used for parallel computing

#分裂样本为N块
#split the triplets into N blocks
function splitBlocks(triplets)
    #每个块的样本量
    num_each = 2000
    total = length(triplets)
    n_blocks::Int = total % num_each == 0 ? floor(Int,total/num_each) : floor(Int,total/num_each)+1
    trs = Dict() 
    for i in 1:n_blocks
        first = (i-1)*num_each+1
        last = i*num_each
        if last>length(triplets)
            last = length(triplets)
        end
        trs[i]=triplets[first:last]
    end
    for (k,ts) in trs
        println("Block number:",k,"Triplets number:",length(ts))
    end
    return trs
end



#优化W步，即针对分裂后的每个数据块进行优化
#optimization of W step in ADMM, i.e., the local optimization based on each data block.
function optimizeW(w, z, y,rho,regWeight,triplets)
    tau = 10.0
    tau2 = 10.0
    punishment_mu = 5000
    n = length(triplets)
    m = length(z)
    println("Triplet number:", n)
    A, b, c = DiagDml.create_coefficients_with_triplets(triplets, Float32(punishment_mu),"huber2",Float32(tau))
    # DiagDml是按照等式约束进行的封装，求解器只要求>=约束，因此将剩余变量删除
    A = A[:,1:m+n]
    c = c[1:m+n]
    
    # 将ADMM惩罚变换为线性规划的约束
    # |w_i - zy_i| <= tau2/rho ==> w_i >= -tau2/rho + zy, - w_i >= -tau2/rho - zy 
    zy = z-y
    b21 = -tau2/rho .+ zy
    b22 = -tau2/rho .- zy
    b = cat(b,b21,b22; dims = 1)
    cols_A = size(A,2)
    rows_A = size(A,1)
    A21 = zeros(m,cols_A)
    A22 = zeros(m,cols_A)
    # 设置新的约束系数
    for i in 1:m
        A21[i,i] = 1
        A22[i,i] = -1
    end
    A2 = vcat(A21,A22)
    # 新加约束为防止无解，新加松驰变量系数
    A3 = ones(2*m,2*m)
    A2 = hcat(A2,A3)
    # 原A矩阵向右扩0，以适应新加松驰变量
    A12 = zeros(rows_A, 2*m)
    A = hcat(A, A12)
    # 原约束系数矩阵和新的约束系数合并
    A = vcat(A,A2)
    c2 = punishment_mu*ones(2*m)
    c = cat(c,c2; dims=1)
    # println(c)
    # println(A)
    # println(b)
    # println(regWeight)
    try
        wAll = RereDmlLpSolver.solveDmlLp(Float32.(c), Float32.(A), Float32.(b),regWeight)
        w = wAll[1:m]
    catch e
        
    end

    # println("A sub-task finished...,w:",w)
    return w
end


#计算L1正则项的数值
function sign_for_l1(z)
    param_c = 1.0E-10 # 1范数拟合函数参数
    y = zeros(length(z))
    indices = abs.(z) .< param_c
    y[indices] = z[indices] .^2 ./(2*param_c) .+ param_c/2.0
    ind1 = z .>= param_c
    y[ind1] .= z[ind1]
    ind2=z .<= param_c
    y[ind2] .= -z[ind2]
    return sum(y)
  end

  # 计算L1正则项的梯度
  function sign_for_l1_gradient(z)
    param_c = 1.0E-10 # 1范数拟合函数参数
    gra = zeros(length(z))
    indices = abs.(z) .< param_c
    gra[indices] = gra[indices] ./ param_c
    gra[z .>= param_c] .= 1
    gra[z .<= param_c] .= -1
    return gra
  end

  # 计算DML正则项，包括L1、L2
  function compute_reg_value(z,alpha)
    l1,l2=alpha,(1-alpha)
    obj_p = l2*sum(z .^ 2) + l1*sign_for_l1(z)
    return obj_p
  end

    # 计算DML正则项的梯度，包括L1、L2
    function compute_reg_grad(z,alpha)
        l1,l2=alpha,(1-alpha)
        grad_p = l2*2.0 * z + l1*sign_for_l1_gradient(z)
        return grad_p
      end

#计算ADMM问题的不一致性L2惩罚
function census_punish_l2(w,z,y)
    v = norm(w-z+y)^2
    return v
end

#计算ADMM问题的不一致性L2惩罚的梯度
function census_punish_l2_grad(w,z,y)
    wy = w+y
    pg = 2*(z-wy) 
    return pg
end


#优化ADMM中的Z步，即汇总步
#Z step in ADMM, which aggreates the paralleled results
function optimizeZ(initZ, w_bar, y_bar,rho,n,regWeight,alpha)
    #原函数
    f(z)=regWeight * compute_reg_value(z,alpha) + n*rho/2* census_punish_l2(w_bar,z,y_bar)
    # f(z)=n*rho/2* census_punish_l2(w_bar,z,y_bar)
    #梯度函数
    function grad(z) 
        return regWeight * compute_reg_grad(initZ,alpha) + n*rho/2* census_punish_l2_grad(w_bar,z,y_bar)
        # return n*rho/2* census_punish_l2_grad(w_bar,z,y_bar)
    end
    function g!(G, xx)
        G .= grad(xx)
    end
    maxStep=20
    error1=0.0
    error2=0.0
    currentStep = 0
    threshold = 1.0E-6
    residual = 100
    z = initZ
    while residual>threshold && currentStep < maxStep
        currentStep += 1
        #ConjugateGradient  LBFGS
        res=optimize(f, g!,z, LBFGS())
        z = Optim.minimizer(res)
        error2 = census_punish_l2(w_bar,z,y_bar)
        residual = error2-error1
        error1 = error2
        # println("residual in solving optimal Z:",residual,", values:",z)
    end
    return z

end

#进行一轮ADMM优化迭代
#conduct one round of the ADMM iteration
function admmUpdate(trs,w_map,z,y_map,rho,regWeight,alpha)
    ws = []
    ys = []
    # addprocs(6)
    for (k,ts) in trs
        w0 = w_map[k]
        y0 = y_map[k]
        wt = optimizeW(w0,z,y0,rho,regWeight,ts)
        w_map[k] = wt
        push!(ws,wt)
        y_map[k] = y0
        push!(ys,y0)
    end
    w_bar = Statistics.mean(ws)
    # println("w_bar values:",w_bar)
    y_bar = Statistics.mean(ys)
    # println("y_bar values:",y_bar)
    n = length(ws)
    # 更新z
    z = optimizeZ(z,w_bar,y_bar,rho,n,regWeight,alpha)
    for (k,y) in y_map
        y_map[k] = y + w_map[k] -z
    end
    error = census_punish_l2(w_bar,z,y_bar)
    return z,error
end

#进行ADMM优化的反复迭代
#comduct the ADMM iterations, until the optimization finished
function admmIterate(triplets,regWeight=1.0,alpha=0.5)
    rho = 10.0
    w_map = Dict()
    y_map = Dict()
    ins = triplets[1].xi
    
    z = ones(length(ins))*2.0
    itr = 1
    max_itr = 30
    seeds = 1:max_itr
    errors = []
    while itr <= max_itr
        # 保证可重复性实验结果
        Random.seed!(seeds[itr])
        shuffle!(triplets)
        trs = splitBlocks(triplets)
        if itr ==1
            # 初始化y_t和w_t的值
            for (k,v) in trs
                w_map[k] = ones(length(ins))
                y_map[k] = ones(length(ins))*1.5
            end
        end
        z,error = admmUpdate(trs,w_map,z,y_map,rho,regWeight,alpha)
        if error< 1.0E-4
            break
        end
        if rho< 1.0E24
            rho = rho * 10
        end
        itr += 1
        push!(errors,error)
        println("ADMM Iterations:",itr,",error:",error)
    end
    z[z.<0] .= 0
    z = sqrt.(z)
    return z,errors
    
end





end
