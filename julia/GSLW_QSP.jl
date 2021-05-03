using GenericLinearAlgebra
using LinearAlgebra
using SpecialFunctions
using Dates
using PolynomialRoots
using Printf
using FFTW
using PyPlot
using MAT

function GSLW(P, p_P, stop_criteria, R_init, R_max, if_polish = false)
    #------------------------------------------------------------------------------------------------------------
    # Input:
    #     P: Chebyshev coefficients of polynomial P, only need to provide non-zero coefficient.
    #        P should satisfy parity constraint and |P|^2 \le 1
    #     p_P: parity of P, 0 -- even, 1 -- odd
    #     stop_eps: algorithm will stop if it find factors that approximate polynomials with error less than
    #               stop_eps on Chebyshev points
    #     R_init: number of bits used at the beginning
    #     R_max: number of bits available
    #     if_polish: whether polish the roots, it cost more time but may improve performance
    #
    # Output:
    #    Phase factor Phi such that real part of (0,0)-component of U_\Phi(x) approximates P(x),
    #    where U_\Phi(x) = e^{\I \phi_0 \sigma_z} \prod_{j=1}^{\qspdeg} \left[ W(x) e^{\I \phi_j \sigma_z} \right]
    #
    #    Besides, the algorithm will return the L^∞ error of such approximation on Chebyshev points
    #
    #------------------------------------------------------------------------------------------------------------
    #
    # Reference:
    #     A. Gily ́en, Y. Su, G. H. Low, and N. Wiebe.
    #     Quantum singular value transformation and beyond: exponentialimprovements for quantum matrix arithmetics.
    #
    # Author: X.Meng, Y. Dong
    # Version 1.0 .... 02/2020
    #
    #------------------------------------------------------------------------------------------------------------
    
    # Step 1: Find all the roots of 1-P^2
    
    R = R_init 
    while(true)
        if(R>=R_max)
            return Inf,[]
        end
        setprecision(BigFloat, R)
        P = big.(P)
        degree = length(P)
        
        if(p_P==0)
            coef_PQ = zeros(BigFloat,degree*2)
            for j=1:degree
                for k=1:degree
                    coef_PQ[j+k-1] -= P[j]*P[k]
                end
            end
            coef_PQ[1] += 1
            coef_PQ1 = zeros(BigFloat,degree)
            for j=1:degree
                coef_PQ1[j] -= P[j]
            end
            coef_PQ1[1] += 1
            coef_PQ2 = zeros(BigFloat,degree)
            for j=1:degree
                coef_PQ2[j] += P[j]
            end
            coef_PQ2[1] += 1
            Proot1 = roots(coef_PQ1, polish = if_polish, epsilon = big.(0.0))
            Proot2 = roots(coef_PQ2, polish = if_polish, epsilon = big.(0.0))
            Proot = [Proot1;Proot2]
        else
            coef_PQ = zeros(BigFloat,degree*2)
            for j=1:degree
                for k=1:degree
                    coef_PQ[j+k] -= P[j]*P[k]
                end
            end
            coef_PQ[1] += 1
            Proot = roots(coef_PQ, polish = if_polish, epsilon = big.(0.0))
        end
        
    # Step 2: Find root of 1-P^2, construct full P and Q by FFT
        
        # recover full root list
        
        all_root = zeros(Complex{BigFloat},length(Proot)*2)
        for i=1:length(Proot)
            tmpnorm = norm(Proot[i])
            tmpangle = angle(Proot[i])
            all_root[2*i-1] = sqrt(tmpnorm)*exp(1im*tmpangle/2)
            all_root[2*i] = -sqrt(tmpnorm)*exp(1im*tmpangle/2)
        end
        
        # Construct W such that W(x)W(x)^*=1-P^2(x)
        
        eps = 1e-16
        S_0 = 0
        S_1 = 0
        S_2 = 0
        S_3 = 0
        S_4 = 0
        S1_list = zeros(Complex{BigFloat},length(all_root))
        S2_list = zeros(Complex{BigFloat},length(all_root))
        S3_list = zeros(Complex{BigFloat},length(all_root))
        S4_list = zeros(Complex{BigFloat},length(all_root))
        for i=1:length(all_root)
            if(abs(all_root[i])<eps)
                S_0 += 1
                continue
            end
            if(abs(imag(all_root[i]))<eps&&real(all_root[i])>0)
                if(real(all_root[i])<1-eps)
                    S_1 += 1
                    S1_list[S_1] = real(all_root[i])
                else
                    S_2 += 1
                    S2_list[S_2] = findmax([real(all_root[i]),1])[1]
                end
                continue
            end
            if(abs(real(all_root[i]))<eps&&imag(all_root[i])>0)
                S_3 += 1
                S3_list[S_3] = all_root[i]
                continue
            end
            if(imag(all_root[i])>0&&real(all_root[i])>0)
                S_4 += 1
                S4_list[S_4] = all_root[i]
            end
        end
        K = abs(P[end])
        
        function get_w(x,use_real = true) # W(x)
            x = big.(x)
            Wx = K*x^(S_0/2)
            eps3 = 1e-24
            if(x==1) # if x==\pm 1, silghtly move x such that make \sqrt{1-x^2}>0
                x -= eps3 
            elseif(x==-1)
                x += eps3
            end
            for i=1:S_1
                Wx *= sqrt(x^2-S1_list[i]^2)
            end
            for i=1:S_2
                Wx *= sqrt(S2_list[i]^2-big.(1))*x+im*S2_list[i]*sqrt(big.(1)-x^2)
            end
            for i=1:S_3
                Wx *= sqrt(abs(S3_list[i])^2+big.(1))*x+im*abs(S3_list[i])*sqrt(big.(1)-x^2)
            end
            for i=1:S_4
                tmpre = real(S4_list[i])
                tmpim = imag(S4_list[i])
                tmpc = tmpre^2+tmpim^2+sqrt(2*(tmpre^2+1)*tmpim^2+(tmpre^2-1)^2+tmpim^4)
                Wx *= tmpc*x^2-(tmpre^2+tmpim^2)+im*sqrt(tmpc^2-1)*x*sqrt(big.(1)-x^2)
            end
            if(use_real)
                return real(Wx)
            else
                return imag(Wx)/sqrt(big.(1)-x^2)
            end
        end
        
        function get_p(x) # P(x)
            P_t = big.(0)
            for j=1:length(P)
                if(p_P==1)
                    P_t += P[j]*x^big.(2*j-1)
                else
                    P_t += P[j]*x^big.(2*j-2)
                end 
            end
            return P_t
        end
        
        # Represent full P and Q under Chevyshev basis
        
        get_wr(x) = get_w(x,true)
        get_wi(x) = get_w(x,false)
        DEG = 2^ceil(Int,log2(degree)+1)
        coef_r = ChebyExpand(get_wr, DEG)
        coef_i = ChebyExpand(get_wi, DEG)
        coef_p = ChebyExpand(get_p, DEG)
        if(p_P==0)
            P_new = 1im.*coef_r[1:2*degree-1]+coef_p[1:2*degree-1]
            Q_new = coef_i[1:2*degree-2].*1im
        else
            P_new = 1im.*coef_r[1:2*degree]+coef_p[1:2*degree]
            Q_new = coef_i[1:2*degree-1].*1im
        end
        
    # Step 3: Get phase factors and check convergence
        
        phi = get_factor(P_new,Q_new)
        max_err = 0
        t = cos.(collect(1:2:(2*degree-1))*big.(pi)/big.(4)/big.(degree))
        for i=1:length(t)
            targ, ret = QSPGetUnitary(phi, t[i])
            P_t = big.(0)
            for j=1:degree
                if(p_P==1)
                    P_t += P[j]*t[i]^big.(2*j-1)
                else
                    P_t += P[j]*t[i]^big.(2*j-2)
                end 
            end
            t_err = norm(real(ret[1,1])-P_t)
            if(t_err>max_err)
                max_err = t_err
            end
        end
        @printf("For degree N = %d, precision R = %d, the estimated inf norm of err is %5.4e\n",length(phi)-1,R,max_err)
        if(max_err<stop_criteria)
            return max_err,phi
        else
            @printf("Error is too big, increase R.\n")
        end
        R = R*2
    end
end

function get_factor(P,Q)
    # From polynomials P, Q generate phase factors phi such that
    # U_\Phi(x) = [P & i\sqrt{1-x^2}Q \\ i\sqrt{1-x^2}Q^* & P]
    # phase factors are generated via a reduction procedure under Chebyshev basis
    phi = zeros(BigFloat,length(P))
    lenP = length(P)
    for i=1:lenP-1
        P, Q, phit = ReducePQ(P, Q)
        phi[end+1-i] = real(phit)
    end
    phi[1] = angle(P[1])
    return phi
end

function ReducePQ(P, Q)
    # A single reduction step
    P = big.(P)
    Q = big.(Q)
    colP = length(P)
    colQ = length(Q)
    degQ = colQ-1
    
    tmp1 = zeros(Complex{BigFloat},colP+1)
    tmp2 = zeros(Complex{BigFloat},colP+1)
    tmp1[2:end] = big.(0.5)*P
    tmp2[1:end-2] = big.(0.5)*P[2:end]
    Px = tmp1 + tmp2
    Px[2] = Px[2] + big.(0.5)*P[1]
    
    if(degQ>0)
        tmp1 = zeros(Complex{BigFloat},colQ+2)
        tmp2 = zeros(Complex{BigFloat},colQ+2)
        tmp3 = zeros(Complex{BigFloat},colQ+2)
        tmp1[1:end-2] = big.(0.5)*Q
        tmp2[3:end] = -big.(1)/big.(4)*Q
        tmp3[1:end-4] = -big.(1)/big.(4)*Q[3:end]
        Q1_x2 = tmp1 + tmp2 + tmp3
        Q1_x2[2] = Q1_x2[2] - 1/big.(4)*Q[2]
        Q1_x2[3] = Q1_x2[3] - 1/big.(4)*Q[1]
    else
        Q1_x2 = zeros(Complex{BigFloat},3)
        Q1_x2[1] = big.(0.5)*Q[1]
        Q1_x2[end] = -big.(0.5)*Q[1]
    end
    
    tmp1 = zeros(Complex{BigFloat},colQ+1)
    tmp2 = zeros(Complex{BigFloat},colQ+1)
    tmp1[2:end] = big.(0.5)*Q
    tmp2[1:end-2] = big.(0.5)*Q[2:end]
    Qx = tmp1 + tmp2
    Qx[2] = Qx[2] + big.(0.5)*Q[1];

    if(degQ>0)
        ratio = P[end]/Q[end]*big.(2)
    else
        ratio = P[end]/Q[end]
    end
    phi = big.(0.5)*angle(ratio)
    rexp = exp(-1im*phi)
    Ptilde = rexp * (Px + ratio*Q1_x2)
    Qtilde = rexp * (ratio*Qx - P)
    Ptilde = Ptilde[1:degQ+1]
    Qtilde = Qtilde[1:degQ]

    return Ptilde,Qtilde,phi
end

function QSPGetUnitary(phase, x)
    # Given phase factors Phi and x, yield U_\Phi(x)
    phase = big.(phase)
    Wx = zeros(Complex{BigFloat},2,2)
    Wx[1,1] = x
    Wx[2,2] = x
    Wx[1,2] = sqrt(1-x^2)*1im
    Wx[2,1] = sqrt(1-x^2)*1im
    expphi = exp.(1im*phase)
    ret = zeros(Complex{BigFloat},2,2)
    ret[1,1] = expphi[1]
    ret[2,2] = conj(expphi[1])

    for k = 2:length(expphi)
        temp = zeros(Complex{BigFloat},2,2)
        temp[1,1] = expphi[k]
        temp[2,2] = conj(expphi[k])
        ret = ret * Wx * temp
    end
    targ = real(ret[1,1])
    return targ,ret
end

function ChebyExpand(func, maxorder)
    # Evaluate Chebyshev coefficients of a polynomial of degree at most maxorder 
    M = maxorder
    theta = zeros(BigFloat,2*M)
    for i=1:2*M
        theta[i] = (i-1)*big.(pi)/M
    end
    f = func.(-cos.(theta))
    c = real.(BigFloatFFT(f))
    c = copy(c[1:M+1])
    c[2:end-1] = c[2:end-1]*2
    c[2:2:end] = -copy(c[2:2:end])
    c = c./(big.(2)*big.(M))
    return c
end

function Chebytonormal(coef)
    #Convert Chebyshev basis to normal basis  
    coef = big.(coef)
    coef2 = zeros(BigFloat,length(coef))
    A = zeros(BigFloat,length(coef),length(coef))
    b = zeros(BigFloat,length(coef))
    t = cos.(collect(1:2:(2*length(coef)-1))*big.(pi)/big.(4)/big.(length(coef)))
    t2 = collect(1:2:(2*length(coef)-1))*big.(pi)/big.(4)/big.(length(coef))
    for i=1:length(coef)
        for j=1:length(coef)
            A[i,j] = t[i]^(j-1)
            b[i] += coef[j]*cos((j-1)*t2[i])
        end
    end
    coef2 = A\b
    #@printf("Error is %5.4e\n",norm(A*coef2-b))
    return coef2
end

function BigFloatFFT(x)
    # Perform FFT on vector x
    # This function only works for length(x) = 2^k
    N = length(x);
    xp = x[1:2:end];
    xpp = x[2:2:end];
    if(N>=8)
        Xp = BigFloatFFT(xp);
        Xpp = BigFloatFFT(xpp);
        X = zeros(Complex{BigFloat},N,1);
        Wn = exp.(big.(-2)im*big.(pi)*(big.(0:N/2-1))/big.(N));
        tmp = Wn .* Xpp;
        X = [(Xp + tmp);(Xp -tmp)];
    elseif(N==2)
        X = big.([1 1;1 -1])*x;
    elseif(N==4)
        X = big.([1 0 1 0; 0 1 0 -1im; 1 0 -1 0;0 1 0 1im]*[1 0 1 0;1 0 -1 0;0 1 0 1;0 1 0 -1])*x;
    end
    return X
end

# Test case 1: Hamiltonian simulation
#
# Here we want to approxiamte e^{-i\tau x} by Jacobi-Anger expansion:
# 
# e^{-i\tau x} = J_0(\tau)+2\sum_{k even} (-1)^{k/2}J_{k}(\tau)T_k(x)+2i\sum_{k odd} (-1)^{(k-1)/2}J_{k}(\tau) T_k(x)
#
# We truncate the series up to N = 1.4\tau+log(10^{14}), which gives an polynomial approximation of e^{-i\tau x} with
# accuracy 10^{-14}. Besides, we deal with real and imaginary part of the truncated series seperatly and divide them
# by a constant factor 2 to enhance stability.
#
# parameters
#     stop_eps: desired accuracy
#     tau: the duration \tau in Hamiltonian simulation
#     R_init: number of bits used at the beginning
#     R_max: number of bits available

stop_eps = 1e-12
tau = 100
R_init = 1024
R_max = 1025

#------------------------------------------------------------------

phi1 = []
phi2 = []
for p_P=0:1
    N = ceil.(Int,tau*1.4+log(1e14))
    if(p_P==0)
        setprecision(BigFloat,4096)
        if(mod(N,2)==1)
            N -= 1
        end
        coef = zeros(BigFloat,N+1)
        for i=1:(round(Int,N/2)+1)
            coef[2*i-1] = (-1)^(i-1)*besselj(big.(2.0*(i-1)),tau)
        end
        coef[1] = coef[1]/2
        P = Chebytonormal(coef)
        P = P[1:2:end]
    else
        setprecision(BigFloat,4096)
        if(mod(N,2)==0)
            N += 1
        end
        coef = zeros(BigFloat,N+1)
        for i=1:round(Int,(N+1)/2)
            coef[2*i] = (-1)^(i-1)*besselj(big.(2*i-1),tau)
        end
        P = Chebytonormal(coef)[2:2:end]
    end
    
    start_time = time()
    err,phi = GSLW(P,p_P,stop_eps,R_init,R_max)
    elpased_time = time()-start_time
    @printf("Elapsed time is %4.2e s\n", elpased_time)
end

# Test case 2: Eigenstate filter
#
# Here we want to generate factors for the eigenstate filter function:
# 
# f_n(x,\delta)=\frac{T_n(-1+2\frac{x^2-\delta^2}{1-\delta^2})}{T_n(-1+2\frac{-\delta^2}{1-\delta^2})}.
#
# We divide f_n by a constant factor \sqrt{2} to enhance stability.
#
# Reference: Lin Lin and Yu Tong
#            Solving quantum linear system problem with near-optimal complexity
#
# parameters
#     stop_eps: desired accuracy
#     n, \delta: parameters of f_n
#     R_init: number of bits used at the beginning
#     R_max: number of bits available
#

stop_eps = 1e-12
n = 100
delta = 0.03
R_init = 1024
R_max = 1025

#------------------------------------------------------------------

function f_n(x,n,delta) 
    val = copy(x)
    delta = big.(delta)
    fact = chebyshev(-big.(1)-big.(2)*delta^2/(big.(1)-delta^2),n)
    if(length(x)==1)
        return chebyshev(-big.(1)+big.(2)*(x^2-delta^2)/(big.(1)-delta^2),n)/fact
    else
        for i=1:length(x)
            val[i] = chebyshev(-1+2*(x[i]^2-delta^2)/(1-delta^2),n)/fact
        end
        return val
    end
end

function chebyshev(x,n) # T_n(x)
    if(abs(x)<=1)
        return cos(big.(n)*acos(x))
    elseif(x>1)
        return cosh(big.(n)*acosh(x))
    else
        return big.((-1)^n)*cosh(big.(n)*acosh(-x))
    end
end

# Obtain expansion of f_n under Chebyshev basis via FFT

setprecision(BigFloat,1024)
M = 2*n
theta = range(0, stop=2*pi, length=2*M+1)
theta = theta[1:2*M]
f = f_n(-cos.(theta),n,delta)
c = real(fft(f))
c = c[1:M+1]
c[2:end-1] = c[2:end-1]*2
c[2:2:end] = - c[2:2:end]
c = c / (2*M)
        
setprecision(BigFloat,4096)
P = Chebytonormal(c)[1:2:end]/sqrt(big.(2.0))
p_P = 0
start_time = time()
err,phi = GSLW(P,p_P,stop_eps,R_init,R_max)
elpased_time = time()-start_time
@printf("Elapsed time is %4.2e s\n", elpased_time)

# Test case 3: Matrix inversion
#
# We would like to approximate 1/x over [1/kappa,1] by a polynomial, such polynomial is generated
# by Remez algorithm and the approximation error is bounded by 10^{-6}
#
# parameters
#     stop_eps: desired accuracy
#     kappa: parameters of polynomial approximation
#     R_init: number of bits used at the beginning
#     R_max: number of bits available
#
 
stop_eps = 1e-12
kappa = 20
R_init = 2048
R_max = 2049

#------------------------------------------------------------------
# even approximation
 
# enter your path here
matpath2 = "../QSPPACK/Data/inversex/"

vars = matread(matpath2 * "coef_xeven_" * string(kappa)*"_6"* ".mat")
coef = vars["coef"]
setprecision(BigFloat,4096)
coef2 = zeros(2*length(coef)-1)
coef2[1:2:end] = coef
P = Chebytonormal(coef2)[1:2:end]
p_P = 0
start_time = time()
err,phi1 = GSLW(P,p_P,stop_eps,R_init,R_max)
elpased_time = time()-start_time
@printf("Elapsed time is %4.2e s\n", elpased_time)

# odd approximation

vars = matread(matpath2 * "coef_xodd_" * string(kappa)*"_6"* ".mat")
coef = vars["coef"]
setprecision(BigFloat,4096)
coef2 = zeros(2*length(coef))
coef2[2:2:end] = coef
P = Chebytonormal(coef2)[2:2:end]
start_time = time()
p_P = 1
err,phi = GSLW(P,p_P,stop_eps,R_init,R_max)
elpased_time = time()-start_time
@printf("Elapsed time is %4.2e s\n", elpased_time)
