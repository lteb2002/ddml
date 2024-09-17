
module DiagDml

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
using RereDmlLpSolver
using TripletModule
using RereDiagDmlSolverLangMul
using RereDiagDmlSolverPf2
using RereDiagDmlSolverPf
using RereDmlLpSolverAdmmSplittingColumns
using LinearAlgebra

export solve_diag_dml

#本模块用于为度量学习问题适配合适的传统求解器（非并行化ADMM，ADMM请参看test_ddml_admm）
#This module works as a helper to connect DML problem with its solvers
#This module is not intended to use ADMM solver

function solve_diag_dml(
    triplets::Array{Triplet,1},
    regWeight::Number = 0,
    a::Number = 0.5,
    distance::String = "huber",
    tau::Number = 2^8
)
    punishment_mu = 5000
    n = length(triplets)
    println("Triplet number:", n)
    A, b, c = create_coefficients_with_triplets(triplets, Float32.(punishment_mu),distance,Float32(tau))
    m = length(triplets[1].xj) # number of features
    r = n # number of slack variables
    s = n # number of surplus variables
    x = nothing
    regType = nothing
    if a == 0
        regType = "l2"
    elseif a == 1.0
        regType = "l1"
    else
        regType = "elastic"
    end
    if lowercase(regType) == "l2" || lowercase(regType) == "elastic"
        # x = RereDiagDmlSolverPf.solveDmlLp(c, A, b, regType, regWeight,a)
        # 拉格朗日乘数法中未使用解决不等式的剩余变量
        x = RereDiagDmlSolverLangMul.solveDmlLp(c[1:m+n], A[:,1:m+n], b, regType, regWeight,a)
        # x = RereDiagDmlSolverPf2.solveDmlLp(c[1:m+n], A[:,1:m+n], b, regType, regWeight,a)
        # x = RereDiagDmlSolverPf2.solveDmlLp(c[1:m], A[:,1:m], b, regType, regWeight,a)
        # x = RereDmlLpSolverAdmmSplittingColumns.solveDmlLpWithAdmm(c,A,b,m,r,s,regType,regWeight)
    else
        x = RereDmlLpSolver.solveDmlLp(c, A, b, regWeight)
        # x = RereDiagDmlSolverPf2.solveDmlLp(c[1:m], A[:,1:m], b, regType, regWeight,a)
    end
    if x != nothing
        x = x[1:m]
        x = [e > 0 ? e : 0.0 for e in x]
        x = sqrt.(x)
        x = round.(x;digits=6)
    end
    return x
end

function create_coefficients_with_triplets(
    triplets::Array{Triplet,1},
    mu::Float32 = 1.0E4,
    distance::String = "huber",
    tau::Float32 = 1.0
)
    A, b, c = nothing, nothing, nothing
    n = length(triplets)
    if n > 0
        # 特征数
        m = length(triplets[1].xi)
        # 样本数行，特征数列
        A = zeros(n, m)
        # 特征数
        c = zeros(m)
        # 样本数所代表的约束数
        b = ones(n) * tau
        for (i, t) in pairs(triplets)
            # thres = 1.0/length(t.xi)
            if distance == "huber"
                # println("Huber distance is used.")
                thres = 0.1
                c += [abs(e) < thres ? e^2 / 2.0 : abs(e)*thres - thres^2 / 2.0 for e in (t.xi - t.xj)]
                A[i, :] = ((t.xk - t.xj) .^ 2 - (t.xi - t.xj) .^ 2)
            else
                c += (t.xi - t.xj) .^ 2 .* t.weight
                # c += (t.xi - t.xj) .^ 2
                # 约束函数最终也以带权重的形式以惩罚项加总到了目标函数
                A[i, :] = ((t.xk - t.xj) .^ 2 - (t.xi - t.xj) .^ 2) .* t.weight
                b[i] = b[i]* t.weight
            end
        end
        # c中封装了原特征、DML松驰变量、取消不等式的剩余变量
        c = cat(c, ones(n) .* mu, zeros(n); dims = 1)
        # 同c，A中也封装了三部分
        A = cat(A, Matrix(I, n, n), -Matrix(I, n, n); dims = 2)
    end
    return (Float32.(A), Float32.(b), Float32.(c))
end


function test_lipschitz()
    return RereDiagDmlSolverLangMul.test_lipschitz()
end


end
