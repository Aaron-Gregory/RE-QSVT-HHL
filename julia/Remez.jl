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

function Remez(targetf, parity, degree, xapp, lef = -1, rig = 1, maxiter = 10, sample_size = 20, eps = 1e-12)
    #--------------------------------------------------------------------------------------------------------------
    # Input:
    #     targetf: target funciton
    #     parity: the parity of approximation function, 0 -- even, 1 -- odd, 2 -- no constraint
    #     degree: the degree of approximation polynomial
    #     xapp: initial points to approximate, if provide a single zero, will choose roots of Chebyshev polynomials
    #     lef, rig: left and right endpoints of approximation interval
    #     maxiter: Max iteration 
    #     sample_size: sample size used in Remez algorithm
    #     eps: accuracy of approximation
    #
    # Output:
    #     Expansion of approximation polynomial under Chebyshev basis
    #
    # We note that our implementation does not strictly follow the reference. The algorithm may crush when 
    # the degree is very large or the problem is ill-conditioned.
    #
    #------------------------------------------------------------------------------------------------------------
    #
    # Reference:
    #     E. W. Cheney.
    #     Introduction to approximation theory
    #
    # Author: X. Meng
    # Version 1.0 .... 02/2020
    #
    #------------------------------------------------------------------------------------------------------------
    
    if(xapp==0)
        xapp = big.(cos.(collect(range(pi/2,stop=0,length=degree+1))))
    end
    iter = 0
    eps2 = 1e-18
    comp = ones(degree+1,1)
    comp[2:2:end] .= -1
    while(true)
        iter += 1
        
        # Step1: Find the best approximation on n+1 given points by solving a linear equation
        
        A = zeros(BigFloat,degree+1,degree+1)
        b = zeros(BigFloat,degree+1,1)
        for i=1:degree
            if(parity==0)
                deg = 2*(i-1)
            elseif(parity==1)
                deg = 2*i-1
            else
                deg = i-1
            end
            for j=1:degree+1
                A[j,i] = chebyshev(xapp[j],deg)
            end
        end
        A[:,end] = comp
        for i=1:degree+1
            b[i] = targetf(xapp[i])
        end
        sol = A\b

        # Step2: Find roots of residual function
        
        eps3 = findmin([eps2,1e-4*abs(sol[end])])[1]
        xroot = zeros(BigFloat,degree,1)
        for i=1:degree
            rootl = xapp[i]
            rootr = xapp[i+1]
            pm = (-1)^(i)*sign(sol[end])
            froot(x) = chebyshevfunc(x,sol[1:end-1],parity,-targetf(x))
            rootiter = 0
            xroot[i] =  brent(rootl,rootr,froot,eps3,eps3)
            if(xroot[i]==Inf)
                return
            end
        end
        
        # Step3: In each pair of adjacent roots, find a point x such that the absolute value of 
        #        residual function is maximized. In addition, values of residual function alternate
        #        in sign.
                 
        xappnew = copy(xapp)
        maxtot = abs(sol[end])
        r_max = -Inf
        y_max = -Inf
        
        for i=1:degree+1
            maxapp = abs(sol[end])
            pm = (-1)^(i)*sign(sol[end])
            if(i==1)
                lend = lef
            else
                lend = xroot[i-1]
            end
            if(i==degree+1)
                rend = rig
            else
                rend = xroot[i]
            end
            exh = collect(range(lend,stop=rend,length=sample_size))
            for j=1:length(exh)
                fval = chebyshevfunc(exh[j],sol[1:end-1],parity,-targetf(exh[j]))
                if(abs(fval)>r_max)
                    r_max = abs(fval)
                end
                if(fval*pm>y_max)
                    y_max = fval*pm
                end
                if(fval*pm>maxapp)
                    maxapp = fval*pm
                    maxtot = maximum([maxtot,maxapp])
                    xappnew[i] = exh[j]
                end
            end
        end
        xapp = copy(xappnew)
        
        if(abs(r_max-y_max)>1e-12)
            @printf("Warning: the interpolation points maybe incorrect\n")
        end
        @printf("The %3d-th itertion: previous error is %5.4e, L_inf approximation error is %5.4e\n",iter,sol[end],maxtot)
        if(maxtot<eps||iter>=maxiter||abs((sol[end]-maxtot)/sol[end])<1e-4)
            return sol[1:end-1]
        end        
    end
    end

function chebyshev(x,n) # T_n(x)
    if(abs(x)<=1)
        return cos(n*acos(x))
    elseif(x>1)
        return cosh(n*acosh(x))
    else
        return (-1)^n*cosh(n*acosh(-x))
    end
end

function chebyshevfunc(x,sol,parity,init)
    # Compute the value of a summation of Chebyshev polynomials at x
    y = init
    for i=1:length(sol)
        if(parity==0)
            deg = 2*(i-1)
        elseif(parity==1)
            deg = 2*i-1
        else
            deg = i-1
        end
        y += sol[i]*chebyshev(x,deg)
    end
    return y
end

function brent(a,b,f,tol1,tol2) 
    # brent method for finding roots on a given interval [a,b]
    # f(a)f(b) should be less than 0, tol1, tol2 are stopping criteria
    fa = f(a)
    fb = f(b)
    if(f(a)*f(b)>0)
        println("Error, f(a)f(b)>0")
        return Inf
    end
    if(abs(f(a))<abs(f(b)))
        tmp = a
        tmp2 = fa
        a = b
        fa = fb
        b = tmp
        fb = tmp2
    end
    c = a
    fc = fa
    s = b
    d = 1e-10
    iter = 0
    mflag = true
    while(true)
        iter += 1
        if(iter>1000)
            println("Brent method: reaches max iteration.")
            return Inf
        end
        if(abs(b-a)<tol2||abs(f(s))<tol1)
            return s
        end
        if(fa!=fc&&fb!=fc)
            s = a*fb*fc/((fa-fb)*(fa-fc))+b*fa*fc/((fb-fa)*(fb-fc))+c*fa*fb/((fc-fa)*(fc-fb))
        else
            s = b-fb*(b-a)/(fb-fa)
        end
        if((s>=b)||(s<=(3*a+b)/4)||((abs(s-b)*2>abs(b-c))&&mflag)||((abs(s-b)*2>abs(d-c))&&(!mflag))||(tol1>abs(b-c)&&mflag)||((tol1>abs(d-c))&&(!mflag)))
            s = (a+b)/2
            mflag = true
        else
            mflag = false
        end
        fs = f(s)
        d = c
        c = b
        fc = fb
        if(fa*fs<0)
            b = s
            fb = fs
        else
            a = s
            fa = fs
        end
        if(abs(f(a))<abs(f(b)))
            tmp = a
            tmp2 = fa
            a = b
            fa = fb
            b = tmp
            fb = tmp2
        end
    end
end


d = Remez(x->1/x, 1, 10, 0, 1, 10)

using DelimitedFiles

# open file in append mode
file = open("example_file.txt", "w") 

# writing to a file using write() method 
writedlm(file, d) 

# We need to close the file in order to write the content from the disk to file 
close(file)
