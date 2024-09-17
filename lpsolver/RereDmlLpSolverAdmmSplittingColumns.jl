
module RereDmlLpSolverAdmmSplittingColumns
using LinearAlgebra
using Optim, LineSearches
export solveDmlLpWithAdmm

#本模块主要用于测试对DML优化问题的系数矩阵进行列分裂测试，本质上并不能并行计算
#This module is used to test splitting the colunms (features/fields of the datasets) of the DML matrix, it cannot be used for parallel computing
    function separateVariables(c, A, b, mainVarNum::Int16,slackNum::Int16,surplusNum::Int16)
        varNum = length(c)
        p1MainVarNum = Int16(floor(mainVarNum / 2)) # the main variable number of part1
        p2MainVarNum = mainVarNum - p1MainVarNum# the main variable number of part2
        p1SlackNum = trunc(Int16,slackNum / 2)# the slack variable number of part1
        p2SlackNum = slackNum - p1SlackNum# the slack variable number of part2
        p1SurplusNum = Int16(ceil(surplusNum / 2))# the surplus variable number of part1
        p2SurplusNum = surplusNum - p1SurplusNum# the surplus variable number of part2
        # println(p1MainVarNum)
        # println(p1SurplusNum)

        mainC=c[1:mainVarNum]
        slackC=nothing
        # println(slackNum)
        if(slackNum != 0)
            slackC=c[mainVarNum+1:mainVarNum+slackNum]
        end
        surplusC=c[mainVarNum+slackNum+1:end]

        mainA=A[:,1:mainVarNum]
        slackA = nothing
        if(slackNum != 0)
            slackA=A[:,mainVarNum+1:mainVarNum+slackNum]
        end
        # println("slackA:",slackA)
        surplusA=A[:,mainVarNum+slackNum+1:end]

        # mainB=b[1:mainVarNum]
        # slackB=b[mainVarNum+1:mainVarNum+slackNum]
        # surplusB=b[mainVarNum+slackNum+1:end]

        p1c = (slackNum == 0 ? vcat(mainC[1:p1MainVarNum],surplusC[1:p1SurplusNum]) : vcat(mainC[1:p1MainVarNum],slackC[1:p1SlackNum],surplusC[1:p1SurplusNum]))
        p2c = (slackNum == 0 ? vcat(mainC[p1MainVarNum+1:end],surplusC[p1SurplusNum+1:end]) : vcat(mainC[p1MainVarNum+1:end],slackC[p1SlackNum+1:end],surplusC[p1SurplusNum+1:end]))

        p1A = slackNum == 0 ? hcat(mainA[:,1:p1MainVarNum],surplusA[:,1:p1SurplusNum]) : hcat(mainA[:,1:p1MainVarNum],slackA[:,1:p1SlackNum],surplusA[:,1:p1SurplusNum])
        p2A = slackNum == 0 ? hcat(mainA[:,p1MainVarNum+1:end],surplusA[:,p1SurplusNum+1:end]) : hcat(mainA[:,p1MainVarNum+1:end],slackA[:,p1SlackNum+1:end],surplusA[:,p1SurplusNum+1:end])
        #
        # p1B=hcat([mainB[1:p1MainVarNum],slackB[1:p1SlackNum],surplusB[1:p1SurplusNum]])
        # p2B=hcat([mainB[p1MainVarNum+1:end],slackB[p1SlackNum+1:end],surplusB[p1SurplusNum+1:end]])
        # println("p1c:",p1c)
        # println("p2c:",p2c)
        # println("p1A:",p1A)
        # println("p2A:",p2A)
        # println("b:",b)
        return (p1c,p1A,p2c,p2A,b)
    end

    function  constraint_residual(x1,x2,p1A,p2A,b)
        p1 = p1A * x1
        p2 = p2A * x2
        res = p1 + p2 - b
        # println(res)
        return res
    end

    function punish(x)
        y=[e<0 ? e^2 : 0.0 for e in x]
        # println("x:",x)
        # y = -log.(x)
        # println("-log(x):",y)
        return sum(y)
    end

    function punish_gradient(x)
        y = [e < 0 ? 2.0 * e : 0 for e in x]
        # y = -1.0./x
        # println(y)
        return y
    end

    function total_error1(x1,x2,p1A,p2A,b)
        res = constraint_residual(x1,x2,p1A,p2A,b)
        s1=sum(res.^2)
        s2=punish(x1)
        return s1+s2
    end

    function total_error2(x1,x2,p1A,p2A,b)
        res = constraint_residual(x1,x2,p1A,p2A,b)
        s1=sum(res.^2)
        s2=punish(x2)
        return s1+s2
    end

    function pure_objective(x1,x2,p1c,p2c,alpha)
        # println("beta:",beta)
        y1 = p1c' * x1 + p2c' * x2
        # println("y1:",y1)
        y2 = alpha * sum(x1.^2) + alpha * sum(x2.^2)
        # println("y2:",y2)
        # println("y4:",y4)
        obj = y1+y2
        return obj
    end

    function objective(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        # println("beta:",beta)
        res = constraint_residual(x1,x2,p1A,p2A,b)
        y1 = p1c' * x1 + p2c' * x2
        # println("y1:",y1)
        y2 = alpha * sum(x1.^2) + alpha * sum(x2.^2)
        # println("y2:",y2)
        y3 = beta * punish(x1) + beta * punish(x2)
        # println("y3:",y3)
        y4 = lambda' * res + 0.5 * gamma * sum(res.^2)
        # println("y4:",y4)
        obj = y1+y2 +y3  + y4
        return obj
    end

    function gradient1(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        y1=p1c + 2 * alpha * x1
        # println("y1:",y1)
        y2=2*beta * punish_gradient(x1)
        # println("y2:",y2)
        y3=p1A' * lambda
        # println("y3:",y3)
        y4=gamma * p1A' * constraint_residual(x1,x2,p1A,p2A,b)
        # println("y4:",y4)
        g = y1+y2+y3+y4
        return g
    end

    function gradient2(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        # g = p2c + 2*alpha*x2+2*beta*punish_gradient(x2)+p2A'*lambda+gamma*p2A*constraint_residual(x1,x2,p1A,p2A,b)
        y1=p2c + 2 * alpha * x2
        # println("y1:",y1)
        y2=2*beta*punish_gradient(x2)
        # println("y2:",y2)
        y3=p2A'*lambda
        # println("y3:",y3)
        y4=gamma * p2A' * constraint_residual(x1,x2,p1A,p2A,b)
        # println("y4:",y4)
        g = y1+y2+y3+y4
        return g
    end

    function updateX1(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        function g!(G, xx)
          G .= gradient1(xx,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        end
        f(x)=objective(x,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        lastX = x1
        beta0=beta
        for i in 1:10
            if(beta<1.0e9)
                beta = beta0^i
            end
            sol=optimize(f, g!,x1, LBFGS())
            x1 = Optim.minimizer(sol)
            error = sum((x1-lastX).^2)
            # error = total_error1(x1,x2,p1A,p2A,b)
            # println("x1 error",error)
            # println("x1:",x1)
            if(error<1.0e-6)
                println("x1 is solved in ",i," steps")
                break
            end
            lastX = x1
        end
        # println("x1:",x1)
        return x1
    end

    function updateX2(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        function g!(G, xx)
          G .= gradient2(x1,xx,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        end
        f(x)=objective(x1,x,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        lastX = x2
        beta0=beta
        for i in 1:10
            if(beta<1.0e9)
                beta = beta0^i
            end
            sol=optimize(f, g!,x2, LBFGS())
            x2 = Optim.minimizer(sol)
            error = sum((x2-lastX).^2)
            # error = total_error2(x1,x2,p1A,p2A,b)
            # println("x2 error",error)
            # println("x2:",x2)
            if(error<1.0e-6)
                println("x2 is solved in ",i," steps")
                break
            end
            lastX = x2
        end
        # println("x2:",x2)
        return x2
    end

    function updateLambda(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        lambda = lambda + gamma * constraint_residual(x1,x2,p1A,p2A,b)
        return lambda
    end

    function solveDmlLpWithAdmm(c,A,b,mainVarNum::Integer,slackNum::Integer,surplusNum::Integer,regType::String="l2", regWeight::Number=1.0)
        c = Float32.(c)
        A = Float32.(A)
        b = Float32.(b)
        mainVarNum = Int16(mainVarNum);slackNum = Int16(slackNum);surplusNum = Int16(surplusNum)
        p1c,p1A,p2c,p2A,b = separateVariables(c, A, b, mainVarNum,slackNum,surplusNum)
        x1 = ones(length(p1c))*1.0
        x2 = ones(length(p2c))*1.0
        lambda = ones(length(b))*1.0
        alpha = regWeight
        beta0 = 100.0
        gamma = 10

        maxStep=20
        error=1.0
        currentStep = 1
        beta = beta0
        while error>1.0E-4 && currentStep < maxStep
            # if(beta<1.0e9)
            #     beta = beta0^currentStep
            # end
            println("beta:",beta)
            x1 = updateX1(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
            x2 = updateX2(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
            lambda = updateLambda(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
            res = constraint_residual(x1,x2,p1A,p2A,b)
            error = sum(res.^2)
            println("Admm Step:",currentStep,",Objective value: ",pure_objective(x1,x2,p1c,p2c,alpha),", error:",error)
            currentStep += 1
        end
        obj = objective(x1,x2,lambda,p1c,p1A,p2c,p2A,b,alpha,beta,gamma)
        p1MainVarNum = Int8(floor(mainVarNum / 2))
        x = vcat(x1[1:p1MainVarNum],x2[1:mainVarNum-p1MainVarNum])
        return x
    end

end
