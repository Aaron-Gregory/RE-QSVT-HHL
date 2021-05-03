using GenericLinearAlgebra
using LinearAlgebra
using SpecialFunctions
using Dates
using PolynomialRoots
using MAT
using Printf
using FFTW

function get_QSP_proj(eps,zeta_A,zeta_B,p_A,p_B,stop_eps,R_init = 64, R_max = 2^10, even_root = false, only_a = false,
                      if_polish = false, GSLW_out = true)
    #------------------------------------------------------------------------------------------------------------
    #
    # Input:
    #     eps: rounding precision, usually be set as eps=stop_eps/30
    #     zeta_A, zeta_B: coefficients of Laurent polynomial A and B, only need to provide coefficients with 
    #                     non-negative degree
    #     p_A, p_B: parity of A and B, 0 -- even, 1 -- odd
    #     stop_eps: algorithm will stop if it find a control achieve error less than stop_eps on N-th
    #               roots of unity, where N is degree of A
    #     R_init: number of bits used at the beginning
    #     R_max: number of bits available
    #     even_root: whether 1-A^2-B^2 is an even polynomial
    #     only_a: whether B=0
    #     if_polish: whether polish the roots, it cost more time but may improve performance
    #     
    # Output:
    #    If GSLW_out=0, the algorithm will outout a list of 2*2 matrices {P_j} such that <+|E_0...E_{2n}|+> 
    #                   approximate A(t^2)+iB(t^2) for all t with |t|=1, where E_j(t)=tP_j+(I-P_j)/t
    #    else,          the algorithm will outout a list of factors {phi_j} such that <+|E_0...E_{n}|+> 
    #                   approximate A(t)+iB(t) for all t with |t|=1, where P_j=e^{iZ\phi_j/2}|+><+|e^{-iZ\phi_j/2},
    #                   E_j(t)=tP_j+(I-P_j)/t (Here A and B should be converted from real polynomials P and Q
    #                   satisfying constraints given in GSLW)
    #    Besides, the algorithm will return the L^∞ error of such approximation on N-th roots of unity
    #
    #------------------------------------------------------------------------------------------------------------
    #
    # Reference:
    #     Jeongwan Haah
    #     Product Decomposition of Periodic Functions in Quantum Signal Processing
    #
    # Author: X.M
    # Version 1.0 .... 02/2020
    #
    #------------------------------------------------------------------------------------------------------------
    
    # Step 1: Find rounded polynomials a(z) and b(z) from (1-10eps)A(z) and (1-10eps)B(z)
    
    R = R_init
    while(true)
        if(R>=R_max)
            return Inf,[]
        end
        setprecision(BigFloat, R)
        eps = big.(eps)
        zeta_A = big.(zeta_A)
        zeta_B = big.(zeta_B)
        zeta_A_eps = (big.(1)-big.(10)*eps)*zeta_A
        zeta_B_eps = (big.(1)-big.(10)*eps)*zeta_B
        degree_N = length(zeta_A) # the true degree of A is in fact degree_N-1
        rational_round = eps/big.((degree_N-1))
        
        zeta_a = zeros(Complex{BigFloat},2*degree_N-1,1)
        zeta_b = zeros(Complex{BigFloat},2*degree_N-1,1)
        
        if(norm(zeta_A_eps[1])<rational_round)
            zeta_a[degree_N] = 0
        else
            zeta_a[degree_N] = zeta_A_eps[1]
        end
        if(norm(zeta_B_eps[1])<rational_round)
            zeta_b[degree_N] = 0
        else
            zeta_b[degree_N] = zeta_B_eps[1]
        end
        degree_n = 0
        for i=1:degree_N-1
            if(norm(zeta_A_eps[i+1])<rational_round)
                zeta_a[degree_N+i] = big.(0)
            else
                zeta_a[degree_N+i] = zeta_A_eps[i+1]
                degree_n = i
            end
            if(norm(zeta_B_eps[i+1])<rational_round)
                zeta_b[degree_N+i] = big.(0)
            else
                zeta_b[degree_N+i] = zeta_B_eps[i+1]
                degree_n = i
            end
            if(p_A==1)
                zeta_a[degree_N-i] = -zeta_a[degree_N+i]
            else
                zeta_a[degree_N-i] = zeta_a[degree_N+i]
            end
            if(p_B==1)
                zeta_b[degree_N-i] = -zeta_b[degree_N+i]
            else
                zeta_b[degree_N-i] = zeta_b[degree_N+i]
            end
        end
        
        # calculate the coefficitents of 1-a(z)^2-b(z)^2
        
        zeta_1ab = zeros(Complex{BigFloat},4*degree_N-3,1)
        degree_1ab = 0
        
        for i=0:2*degree_N-2
            coff = big.(0)
            for j=i+1:2*degree_N-1
                coff -= zeta_a[j]*zeta_a[2*degree_N+i-j]
                coff -= zeta_b[j]*zeta_b[2*degree_N+i-j]
            end
            if(norm(coff)!=0)
                degree_1ab = i
            end
            zeta_1ab[2*degree_N-1+i] = real(coff)
            zeta_1ab[2*degree_N-1-i] = real(coff)
        end
        zeta_1ab[2*degree_N-1] += 1
        zeta_1ab = copy(zeta_1ab[2*degree_N-1-degree_1ab:2*degree_N-1+degree_1ab])
        
    # Step2: Find all roots of 1-a(z)^2-b(z)^2 with accuarcy R, 
    #        we use Julia's internal rountine to find all roots
     
        # when 1-a(z)^2-b(z)^2 has special structure, we may reduce provlem scale
        
        if(only_a) 
            zeta_1a1 = copy(zeta_a[degree_N-degree_n:degree_N+degree_n])
            zeta_1a1[degree_n+1] -= 1.0
            zeta_1a2 = copy(zeta_a[degree_N-degree_n:degree_N+degree_n])
            zeta_1a2[degree_n+1] += 1.0
            if(even_root)
                zeta_1a1 = copy(zeta_1a1[1:2:end])
                zeta_1a2 = copy(zeta_1a2[1:2:end])
            end
            all_root1 = roots(-zeta_1a1, polish = if_polish, epsilon = big.(0.0))
            all_root2 = roots(zeta_1a2, polish = if_polish, epsilon = big.(0.0))
            
            if(even_root)
                all_root = zeros(Complex{BigFloat}, 2*(length(all_root1)+length(all_root2)))
                for i=1:length(all_root1)
                    tmpnorm = norm(all_root1[i])
                    tmpangle = angle(all_root1[i])
                    all_root[2*i-1] = sqrt(tmpnorm)*exp(1im*tmpangle/2)
                    all_root[2*i] = -sqrt(tmpnorm)*exp(1im*tmpangle/2)
                end
                tmp_len = 2*length(all_root1)
                for i=1:length(all_root2)
                    tmpnorm = norm(all_root2[i])
                    tmpangle = angle(all_root2[i])
                    all_root[tmp_len+2*i-1] = sqrt(tmpnorm)*exp(1im*tmpangle/2)
                    all_root[tmp_len+2*i] = -sqrt(tmpnorm)*exp(1im*tmpangle/2)
                end
            else
                all_root = [all_root1;all_root2]
            end
        elseif(even_root)
            zeta_1ab2 = copy(zeta_1ab[1:2:end])
            all_rootmp = roots(zeta_1ab2, polish = if_polish, epsilon = big.(0.0))
            all_root = zeros(Complex{BigFloat}, 2*length(all_rootmp))
            for i=1:length(all_rootmp)
                tmpnorm = norm(all_rootmp[i])
                tmpangle = angle(all_rootmp[i])
                all_root[2*i-1] = sqrt(tmpnorm)*exp(1im*tmpangle/2)
                all_root[2*i] = -sqrt(tmpnorm)*exp(1im*tmpangle/2)
            end
        else
            all_root = roots(zeta_1ab, polish = if_polish, epsilon = big.(0.0))
        end

    # Step3: Evaluate complementary polynomials c(z) and d(z) on points {exp(2*k*pi*im/D)|k=1,2,...},
    #        where D is a power of 2 such that D>2n+1. These values will be utilized to perform FFT.
        
        # find roots with norm less than 1
        
        if(if_polish)  
            eps_root = 1e-16
            root_list = zeros(Complex{BigFloat},4*degree_1ab,1)
            count = 0
            for i=1:length(all_root)
                norm_root = norm(all_root[i])
                if(abs(norm_root-1)<eps/(4*degree_N^2))
                    @printf("Warning: a root has norm close to 1\n")
                end
                if(norm_root<big.(1))
                    if(count<=degree_1ab)
                        flag = 0
                        for j=1:count
                            if(norm(root_list[j]-all_root[i])<eps_root)
                                flag = 1
                                break
                            end
                        end
                        if(flag==1)
                            continue
                        end
                    
                        if(abs(all_root[i])<eps_root)
                            count += 1  
                            root_list[count] = all_root[i]
                            continue
                        end
                        if(abs(imag(all_root[i]))<eps_root)
                            count += 2  
                            root_list[count-1] = all_root[i]
                            root_list[count] = -all_root[i]
                            continue
                        end
                        if(abs(real(all_root[i]))<eps_root)
                            count += 2  
                            root_list[count-1] = all_root[i]
                            root_list[count] = conj(all_root[i])
                            continue
                        end
                        count += 4
                        root_list[count-3] = all_root[i]
                        root_list[count-2] = conj(all_root[i])
                        root_list[count-1] = -all_root[i]
                        root_list[count] = conj(-all_root[i])
                    else
                        break
                    end
                end
            end
        else
            root_list = zeros(Complex{BigFloat},4*degree_1ab,1)
            count = 0
            for i=1:length(all_root)
                norm_root = norm(all_root[i])
                if(abs(norm_root-1)<eps/(4*degree_N^2))
                    @printf("Warning: a root has norm close to 1\n")
                end
                if(norm_root<big.(1))
                    count += 1
                    root_list[count] = all_root[i]
                end
            end
        end
        root_list = root_list[1:degree_1ab]
        count = count+1
        
        # find alpha, which is required in order to construct c(z) and d(z)
        
        z = big.(1)
        e_1 = big.(1)
        e_2 = big.(1)
        for j=1:degree_1ab
            e_1 *= (z-root_list[j])
            e_2 *= (big.(1)/z-root_list[j])
        end
        
        alpha = sum(zeta_1ab)/(e_1*e_2)
        alpha = real(alpha)
        
        if((alpha==NaN)||(alpha==Inf)||(alpha<0))
            @printf("Alpha is not correct, increase R\n")
            R = R*2
            continue
        end
        
        if(count<=degree_1ab)
            @printf("Number of roots is incorrect, increase R\n")
            R = R*2
            continue
        end
        
        # calculate value on D points
        
        D = 2^(ceil(Int,log2(2*degree_n+1)))
        cvalue_list = big.(zeros(Complex{BigFloat},D,1))
        dvalue_list = big.(zeros(Complex{BigFloat},D,1))
        for i=0:D-1
            z = exp(big.(2)*big.(pi)*im*big.(i)/big.(D))
            z_ = big.(1)/z

            e_1 = z_^floor(Int,degree_1ab/2)
            e_2 = z^floor(Int,degree_1ab/2)
            for j=1:degree_1ab
                e_1 *= (z-root_list[j])
                e_2 *= (z_-root_list[j])
            end
            dvalue_list[i+1] = sqrt(alpha)*(e_1+e_2)/big.(2)
            cvalue_list[i+1] = sqrt(alpha)*(e_1-e_2)/big.(-2)*big.(1)im
        end

    # Step 4: Compute the discrete fast Fourier transform of the function F(z) = a(z)I+b(z)iX+
    #         c(z)iY+d(z)iZ, where X,Y,Z are pauli matrices
        
        # calculate value on D points

        avalue_list = big.(zeros(Complex{BigFloat},D,1))
        bvalue_list = big.(zeros(Complex{BigFloat},D,1))
        for i=0:D-1
            z = exp(big.(2)*big.(pi)*im*big.(i)/big.(D))
            a_tmp = big.(0)
            b_tmp = big.(0)
            for j=1:2*degree_N-1
                a_tmp += z^(j-degree_N)*zeta_a[j]
                b_tmp += z^(j-degree_N)*zeta_b[j]
            end
            avalue_list[i+1] = a_tmp
            bvalue_list[i+1] = b_tmp
        end
        
        # perform FFT on each component
        
        C = zeros(Complex{BigFloat},2,2,2*degree_n+1)
        C_11 = BigFloatFFT(avalue_list+im*dvalue_list)
        C_12 = BigFloatFFT(cvalue_list+im*bvalue_list)
        C_21 = BigFloatFFT(-cvalue_list+im*bvalue_list)
        C_22 = BigFloatFFT(avalue_list-im*dvalue_list)
        C[1,1,:] = [C_11[D-degree_n+1:D];C_11[1:degree_n+1]]/big.(D)
        C[1,2,:] = [C_12[D-degree_n+1:D];C_12[1:degree_n+1]]/big.(D)
        C[2,1,:] = [C_21[D-degree_n+1:D];C_21[1:degree_n+1]]/big.(D)
        C[2,2,:] = [C_22[D-degree_n+1:D];C_22[1:degree_n+1]]/big.(D)
        
    # Step 5: From matrix C, caluculate the projection matrix P (and thus our objective E)
        
        if(GSLW_out)
            C = C[:,:,1:2:end]
            phi = zeros(Float64,degree_n+1)
            P = zeros(Complex{BigFloat},2,2,degree_n+1)
            for i=1:degree_n
                tmp_P = C[:,:,end]'*C[:,:,end]
                tmp_Q = C[:,:,1]'*C[:,:,1]
                P_i = tmp_P/tr(tmp_P)
                Q_i = tmp_Q/tr(tmp_Q)
                P[:,:,degree_n+2-i] = P_i
                phi[degree_n+2-i] = getarg(P[1,2,degree_n+2-i])
                C_new = zeros(Complex{BigFloat},2,2,degree_n+1-i)
                for j=1:degree_n+1-i
                    C_new[:,:,j] = C[:,:,j]*Q_i+C[:,:,j+1]*P_i
                end
                C = copy(C_new)
            end
            P[:,:,1] = copy(C[:,:,1])
            phi[1] = getarg(P[1,1,1])
        
        
            # Step 6: Check if phi is accurate enough, if so output phi, else increase R    
        
            max_err = 0
            zeta_A = big.(zeta_A)
            zeta_B = big.(zeta_B)
            for i=1:degree_N
                t = exp(im*big.(pi)*big.(2*i)/big.(degree_N))
                QSP_value = getQSPvalue(P,t)
                true_value = big.(0)
                for j=0:length(zeta_A)-1
                    true_value += t^(j)*zeta_A[j+1]
                    true_value += im*t^(j)*zeta_B[j+1]
                    if(j!=0)
                        if(p_A==0)
                            true_value += t^(-j)*zeta_A[j+1]
                        else
                            true_value -= t^(-j)*zeta_A[j+1]
                        end
                        if(p_B==0)
                            true_value += im*t^(-j)*zeta_B[j+1]
                        else
                            true_value -= im*t^(-j)*zeta_B[j+1]
                        end 
                    end
                end
                t_err = norm(QSP_value-true_value)
                if(t_err>max_err)
                    max_err = t_err
                end
            end
            @printf("For degree N = %d, precision R = %d, the estimated inf norm of err is %5.4e\n",degree_N-1,R,max_err)
            if(max_err<stop_eps)
                return max_err,phi
            else
                @printf("Error is too big, increase R.\n")
            end
            R = R*2
        else
            P = zeros(Complex{BigFloat},2,2,2*degree_n+1)
            for i=1:2*degree_n
                tmp_P = C[:,:,end]'*C[:,:,end]
                tmp_Q = C[:,:,1]'*C[:,:,1]
                P_i = tmp_P/tr(tmp_P)
                Q_i = tmp_Q/tr(tmp_Q)
                P[:,:,2*degree_n+2-i] = P_i
                C_new = zeros(Complex{BigFloat},2,2,2*degree_n+1-i)
                for j=1:2*degree_n+1-i
                    C_new[:,:,j] = C[:,:,j]*Q_i+C[:,:,j+1]*P_i
                end
                C = copy(C_new)
            end
            P[:,:,1] = copy(C[:,:,1])
        
            # Step 6: Check if P is accurate enough, if so output P, else increase R    
        
            max_err = 0
            zeta_A = big.(zeta_A)
            zeta_B = big.(zeta_B)
            for i=1:degree_N
                t = exp(im*big.(pi)*big.(2*i)/big.(degree_N))
                QSP_value = getQSPvalue(P,t)
                true_value = big.(0)
                for j=0:length(zeta_A)-1
                    true_value += t^(2*j)*zeta_A[j+1]
                    true_value += im*t^(2*j)*zeta_B[j+1]
                    if(j!=0)
                        if(p_A==0)
                            true_value += t^(-2*j)*zeta_A[j+1]
                        else
                            true_value -= t^(-2*j)*zeta_A[j+1]
                        end
                        if(p_B==0)
                            true_value += im*t^(-2*j)*zeta_B[j+1]
                        else
                            true_value -= im*t^(-2*j)*zeta_B[j+1]
                        end 
                    end
                end
                t_err = norm(QSP_value-true_value)
                if(t_err>max_err)
                    max_err = t_err
                end
            end
            @printf("For degree N = %d, precision R = %d, the estimated inf norm of err is %5.4e\n",degree_N-1,R,max_err)
            if(max_err<stop_eps)
                return max_err,P
            else
                @printf("Error is too big, increase R.\n")
            end
            R = R*2
        end
    end
end

function getarg(x)
    # Obtain the angle of a complex number
    x = x/abs(x)
    xr = acos(real(x))
    xi = asin(imag(x))
    if(xi>0)
        if(xr>0)
            return xr-2*pi
        else
            return xr
        end
    else
        return -xr
    end
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

function getQSPvalue(P,t)
    # Given P_j (0\le j \le n), yield <+|E_0...E_{n}|+>, where E_j(t)=tP_j+(I-P_j)/t
    eig_x = Complex{BigFloat}[1;1]/sqrt(big.(2))
    init = copy(eig_x)
    for i=1:size(P)[3]-1
        init = init/t+(t-big.(1)/t)*P[:,:,end+1-i]*init
    end
    return eig_x'*P[:,:,1]*init
end

# Test case 1: Hamiltonian simulation
#
# Here we want to approxiamte e^{-i\tau x} by Jacobi-Anger expansion:
# 
# e^{-i\tau x} = J_0(\tau)+2\sum_{k even} (-1)^{k/2}J_{k}(\tau)T_k(x)+2i\sum_{k odd} (-1)^{(k-1)/2}J_{k}(\tau) T_k(x)
#
# We truncate the series up to N = 1.4\tau+log(10^{14}), which gives an polynomial approximation of e^{-i\tau x} with
# accuracy 10^{-14}. Besides, we deal with real and imaginary part of the truncated series seperatly and divide it
# by a constant factor 2 to enhance stability.
#
# A real polynomial P defined on [-1,1] is converted to input of the Haah method through the following way. Suppose
# P has expansion under Chebyshev basis, P = \sum_{k=0}^n a_kT_k(x), then A is taken as A=a_0+\sum_{k=1}^n a_k(z^k+1/z^k)/2
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

for parity=0:1 
    N = ceil.(Int,tau*1.4.+log(1e14))
    if(parity==0)
        setprecision(BigFloat,1024)
        if(mod(N,2)==1)
            N -= 1
        end
        zeta_A = zeros(Complex{BigFloat},1,N+1)
        zeta_B = zeros(Complex{BigFloat},1,N+1)
        p_A = 0
        p_B = 0
        for kk=1:(round(Int,N/2)+1)
            zeta_A[2*kk-1] = (-1)^(kk-1)*besselj(big.(2.0*(kk-1)),tau)/2.0
        end
    else
        setprecision(BigFloat,1024)
        if(mod(N,2)==0)
            N += 1
        end
        zeta_A = zeros(Complex{BigFloat},1,N+1)
        zeta_B = zeros(Complex{BigFloat},1,N+1)
        p_A = 0
        p_B = 0
        for kk=1:round(Int,(N+1)/2)
            zeta_A[2*kk] = (-1)^(kk-1)*besselj(big.(2.0*kk-1),tau)/2.0
        end
    end
    eps = stop_eps/30
    start_time = time()
    
    if(parity==0)
        err, phi1 = get_QSP_proj(eps,zeta_A,zeta_B,p_A,p_B,stop_eps,R_init,R_max,true,true)
    else
        err, phi2 = get_QSP_proj(eps,zeta_A,zeta_B,p_A,p_B,stop_eps,R_init,R_max,false,true)
    end
    
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
# A real polynomial P defined on [-1,1] is converted to input of the Haah method through the following way. Suppose
# P has expansion under Chebyshev basis, P = \sum_{k=0}^n a_kT_k(x), then A is taken as A=a_0+\sum_{k=1}^n a_k(z^k+1/z^k)/2
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
R_max = 2048

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

setprecision(BigFloat,1024)

# Obtain expansion of f_n under Chebyshev basis via FFT

M = 2*n
theta = range(0, stop=2*pi, length=2*M+1)
theta = theta[1:2*M]
f = f_n(-cos.(theta),n,delta)
c = real(fft(f))
c = c[1:M+1]
c[2:end-1] = c[2:end-1]*2
c[2:2:end] = - c[2:2:end]
c = c / (2*M)
        
zeta_A = zeros(Complex{BigFloat},1,length(c))
zeta_B = zeros(Complex{BigFloat},1,length(c))
zeta_A[1] = c[1]/sqrt(big.(2.0))
for k=2:length(c)
    if(mod(k,2)==1)
        zeta_A[k] = c[k]/sqrt(big.(8.0))
    end
end
p_A = 0
p_B = 0
eps = stop_eps/30
start_time = time()
err,phi = get_QSP_proj(eps,zeta_A,zeta_B,p_A,p_B,stop_eps,R_init,R_max,true,true)
elpased_time = time()-start_time
@printf("Elapsed time is %4.2e s\n", elpased_time)

# Test case 3: Matrix inversion
#
# We would like to approximate 1/x over [1/kappa,1] by a polynomial, such polynomial is generated
# by Remez algorithm
#
# A real polynomial P defined on [-1,1] is converted to input of the Haah method through the following way. Suppose
# P has expansion under Chebyshev basis, P = \sum_{k=0}^n a_kT_k(x), then A is taken as A=a_0+\sum_{k=1}^n a_k(z^k+1/z^k)/2
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

eps = stop_eps/30

# enter your path here
matpath2 = "Data\\inversex\\"
vars = matread(matpath2 * "coef_xeven_" * string(kappa)*"_6"* ".mat")
coef = vars["coef"]
zeta_A = zeros(Complex{BigFloat},1,2*length(coef)-1)
zeta_B = zeros(Complex{BigFloat},1,2*length(coef)-1)
for kk=1:length(coef)
    zeta_A[2*kk-1] = coef[kk]/2
end
zeta_A[1] *= 2
p_A = 0
p_B = 0
start_time = time()
err,phi1 = get_QSP_proj(eps,zeta_A,zeta_B,p_A,p_B,stop_eps,R_init,R_max,true,true)
elpased_time = time()-start_time
@printf("Elapsed time is %4.2e s\n", elpased_time)

# odd approximation

vars = matread(matpath2 * "coef_xodd_" * string(kappa)*"_6"* ".mat")
coef = vars["coef"]
zeta_A = zeros(Complex{BigFloat},1,2*length(coef))
zeta_B = zeros(Complex{BigFloat},1,2*length(coef))
for kk=1:length(coef)
    zeta_A[2*kk] = coef[kk]/2
end
p_A = 0
p_B = 0
start_time = time()
err,phi2 = get_QSP_proj(eps,zeta_A,zeta_B,p_A,p_B,stop_eps,R_init,R_max,false,true)
elpased_time = time()-start_time
@printf("Elapsed time is %4.2e s\n", elpased_time)
