# Test case: Matrix inversion
#
# We would like to approximate 1/x over [1/kappa,1] by a polynomial (even or odd)
#
# You may change the target function to approximate desired function, and you can
# save coefficients of approximation polynomail in .mat file and solve for corresponding
# phase factors via optimization method.
#
# parameters
#     kappa: parameters of polynomial approximation
#     degree: degree of freedom of approximation polynomial (not the degree)
#     stop_eps: desired accuracy
#     parity: parity of the approximation polynomial
#     R_high: number of bits sued in high-precision arithmetic
#     save_mat: whether to save the coefficient as a file (.mat)
#     where_save: path to save the data file
#     save_name: name of the data file

using GenericLinearAlgebra
using LinearAlgebra
using SpecialFunctions
using Dates
using MAT
using Roots
using FFTW
using PyPlot
using PolynomialRoots
using Printf


kappa = 10
degree = 60
stop_eps = 1e-6
parity = 0
R_high = 512
save_mat = true
where_save = ""
save_name = ""

#------------------------------------------------------------------

function inversex(x) #1/x divided by a constant factor
    return big.(1)/(big.(4)*kappa*x)
end

setprecision(BigFloat,R_high)
xapp = big.(cos.(collect(range(pi/2,stop=0,length=degree+1)))*(kappa-1)/kappa.+(1/kappa))
solu = Remez(inversex, parity, degree, xapp, big.(1)/kappa, 1.0, 20, 20, stop_eps)
solu = Float64.(solu)

if(save_mat)
    matpath = where_save*"Data\\"
    if(save_name!="")
        mattest = matopen(matpath * save_name * ".mat","w")
    else
        mattest = matopen(matpath * "coef_x_" * string(kappa) * "_" * string(ceil(Int,-log10(stop_eps))) * ".mat","w")
    end
    write(mattest,"coef",Float64.(solu))
    write(mattest,"parity",parity)
    close(mattest)
end