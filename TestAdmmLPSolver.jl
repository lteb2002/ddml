

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./lpsolver")
using RereDmlLpSolverAdmm
using RereDmlLpSolverPf

"""
The variables in the example consist of three types:
the first four are the main variables corresponding to the features of the dataset;
the second four are the slack variables ;
the third four are the surplus variables;
For the details of the model, please refer to the paper 'A Fast Diagonal Distance Metric Learning Approach for Large-Scale Datasets', section 4.3
"""

c=Vector([-2, -4, -1,-1,   5,5,5,5,   0,0, 0,0])
b= Vector([8, 6, 6,9])
A= Matrix([ 1.0  3.0  0.0  4.0     1  0  0  0     -1  0  0  0;
            2.0  1.0  0.0  0.0     0  1  0  0      0 -1  0  0;
            0.0  1.0  1.0  1.0     0  0  1  0      0  0 -1  0;
            1.0  1.0  1.0  0.0     0  0  0  1      0  0  0 -1])
@time x = RereDmlLpSolverAdmm.solveDmlLpWithAdmm(c,A,b,4,4,4,"l2",10)
println("Admm solution",x[1:4])

"""
solve the same problem with Pelnalty Function method and compare the result...
"""

@time x=RereDmlLpSolverPf.solveDmlLp(c,A,b,"l2",10)
println("Pf solution",x[1:4])
