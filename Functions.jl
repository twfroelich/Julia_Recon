using MIRT: ir_mri_sensemap_sim, ellipse_im
using CUDA
using CUDA.CUSPARSE

# Recon stuff
export Calc_Lipshitz
function Calc_Lipshitz(Enc)
    Z = randn(eltype(Enc),size(Enc,2))
    @inbounds for i in 1:50
        Z1 = Enc'*(Enc*vec(Z))
        Z = Z1 / norm(Z1, 2)
    end
    return 2 * norm(Enc'*(Enc*Z))
end
function Enc_reshape(Enc)
    @views begin
        Enc = permutedims(Enc, [3,1,2])
        szEnc = size(Enc)
        m = collect(szEnc[2:end])
        Enc = reshape(Enc, szEnc[1], prod(szEnc[2:end])); # change Enc into a matrix
    end
    return Enc, m
end
function Signal_coils(Encoding_matrix, nobj;
    ncoils=1,
    noise_scale=0,
    coil_dist=10,
    T=Float32,
)
    @inbounds @views begin
        Object_original = eltype(T).(ellipse_im(nobj[1], nobj[2]))
        if ncoils .== 1
            Signal_new = reshape(reshape(Object_original, 1, :) * Encoding_matrix, :)
            Signal_new = Signal_new .+ (noise_scale * randn(Complex{eltype(T)}, size(Signal_new)))

            Object_coils = Object_original

            return ComplexF32.(Signal_new), Object_coils, Object_original
        else   
            smap = ir_mri_sensemap_sim(dims=nobj, ncoil=ncoils, orbit_start=[90], coil_distance=coil_dist)
            Object_coils = Object_original .* smap

            Signal_new = Vector{Complex{eltype(T)}}(undef, 0)
            for index = 1:ncoils
                Signal = reshape(reshape(Object_coils[:,:,index], 1, :) * Encoding_matrix, :)
                Signal = Signal .+ (noise_scale * randn(Complex{eltype(T)}, size(Signal))) 

                Signal_new = vcat(Signal_new, Signal)
            end
            S_matrix = sparse(Diagonal(selectdim(smap, ndims(smap), 1)[:]))
            for index = 2:ncoils
                S_matrix = vcat(S_matrix, sparse(Diagonal(selectdim(smap, ndims(smap), index)[:])))
            end

            return ComplexF32.(Signal_new), Object_coils, Object_original, S_matrix
        end
    end
    
end
function Coil_matrix(S_matrix, Enc, ncoils)
    @views begin
        out = kron(sparse(I(ncoils)), Enc) * (S_matrix)
    end
    return out
end
export Signal_Enc
function Signal_Enc(Mxy, nobj;
    T=Float32,
    ncoils=1,
    noise_scale=0,
    coil_dist=5,
)
    @views begin
        #Encoding_matrix = reshape(Mxy, prod(nobj), prod(size(Mxy[1,1,:])))
        
        if ndims(Mxy) === 3 
            Encoding_matrix = reshape(Mxy, prod(nobj), prod(size(Mxy[1,1,:])))
        else
            Encoding_matrix = Mxy
        end

        if ncoils === 1
            Signal_new, Object_coils, Object_original = Signal_coils(Encoding_matrix, nobj; ncoils, noise_scale, coil_dist)
            Encoding_matrix = reshape(Encoding_matrix, nobj[1], nobj[2], prod(size(Encoding_matrix[1,:])))
            Enc, ~ = Enc_reshape(Encoding_matrix)

            return Enc, Enc, ComplexF32.(Signal_new), Object_coils, Object_original
        else
            Signal_new, Object_coils, Object_original, S_matrix = Signal_coils(Encoding_matrix, nobj; ncoils, noise_scale, coil_dist)
            Encoding_matrix = reshape(Encoding_matrix, nobj[1], nobj[2], prod(size(Encoding_matrix[1,:])))
            Enc, ~ = Enc_reshape(Encoding_matrix)
            F = Coil_matrix(S_matrix, Enc, ncoils)

            return Enc, F, ComplexF32.(Signal_new), Object_coils, Object_original, S_matrix
        end
    end
end
# ------
# Optimization Algorithms CPU
function gr_restart(Fgrad, ynew_yold, restart_cutoff)
	return sum(Float32, real(-Fgrad .* ynew_yold)) <= restart_cutoff * norm(Fgrad) * norm(ynew_yold)
end
export ir_fista_full
"""
Generic FISTA method with cost calculation
Fast Iterative Shrinkage/Thresholding Algorithm [FISTA]
Modified momentum to give POGM

Adapted by Taylor Froelich; Minnesota 2021
Based off the work done previously by Sydney Williams and Jeff Fessler at U of Michigan

# Input Arguments:
-`x0` initial estimate
-`∇f` function returning gradient:
-`Step` user-selected step size (1/Lipschitz)
-`g_prox` function performing proximal ste
-`Fcost` function calculating cost at each iteration

# Optional Arguments:
-`norm_tol` stopping tolerance for change in relative norm() ||x[n+1]-x[n]||/||x[0]||
-`max_tol` stopping tolerance for change in max abs value ||x[n+1]-x[n]||_inf/||x[0])||
-`design_tol` stopping tolerance for relative norm to target pattern ||x[n+1]-x[n]||/||d.*W||

-`stop` set to 0 to eliminate stopping criterion
        1: norm change 2: max value change 3: norm design change

-`niter`  total number of iterations
-`target` design target pattern
-`weight` mask used to weight target pattern

# Outputs:
-`xs` estimates each iteration
"""
function ir_fista_full(x0,Fcost::Function,∇f::Function,step::Real;
    g_prox::Function=(z, step::Real) -> z,
    norm_tol=1e-3,
    max_tol=1e-5,
    design_tol=1e-4,
    stop=0, 
    niter=50, 
    target=1, 
    weight=1, 
    T=ComplexF32,
)
    #= 
    Generic FISTA method with cost calculation
    Fast Iterative Shrinkage/Thresholding Algorithm [FISTA]
    Modified momentum to give POGM

    Adapted by Taylor Froelich; Minnesota 2021
    Based off the work done previously by Sydney Williams and Jeff Fessler at U of Michigan
 
    Input Arguments:
    x0          [np 1]         initial estimate
    ∇f          [Fun]          function returning gradient:
    Step		[Fun]	       user-selected step size (1/Lipschitz)
    g_prox      [Fun]          function performing proximal ste
    Fcost      [Fun]          function calculating cost at each iteration
 
    Optional Arguments:
    norm_tol    [1]            stopping tolerance for change in relative norm() ||x[n+1]-x[n]||/||x[0]||
    max_tol     [1]            stopping tolerance for change in max abs value ||x[n+1]-x[n]||_inf/||x[0])||
    design_tol  [1]            stopping tolerance for relative norm to target pattern ||x[n+1]-x[n]||/||d.*W||
    
    stop        [1]            (0|1|2|3) set to 0 to eliminate stopping criterion
                                1: norm change 2: max value change 3: norm design change

    niter       [1]            total number of iterations
     
    target      [nz 1]         design target pattern
    weight      [nz 1]         mask used to weight target pattern
 
    Outputs:
 	xs          [np niter]	    estimates each iteration =#
    @views begin
        # xs = zeros(eltype(T),length(x0),Int(niter))
        cost = zeros(Float64, Int(niter))

        x = zeros(ComplexF32, size(x0))
        v = zeros(ComplexF32, size(x0))

        norm_int = norm(x0)
        max_int = norm(x0, Inf)

        x_old = x0
        t_old = 1
        cost_max = Fcost(x0)

        @inbounds for iter = 1:niter
        x .= g_prox(v - step .* ∇f(v), step)

        t = (1 + sqrt(1 + 4 * t_old^2)) / 2
        v .= x + (t_old - 1) / t * (x - x_old) + (t_old / t) * (x - v)

        cost_old = Fcost(x_old)
        norm_change = norm(x_old - x) / norm_int
        max_change = norm(x_old - x, Inf) / max_int
        norm_design = norm(x_old - x) / norm(target .* weight)

        x_old .= x
        t_old = t
            
            # Calculates cost at each iteration
            # xs[:,iter] .= vec(x);
        cost[iter] = Fcost(x);          

            # Stopping criterion
        if stop != 0
            if stop == 1 && norm_change < norm_tol  
                xs = xs[:,1:iter];                        
                break
            elseif stop == 2 && max_change < max_tol
                    xs = xs[:,1:iter]
                    break
                elseif stop == 3 && norm_design < design_tol
                    xs = xs[:,1:iter];    
                    break
            end
        end      
    end
    end
    return vec(x), cost
end
export ir_fista
"""
Generic FISTA method with cost calculation with modifications to momentum to give POGM

Adapted by Taylor Froelich Minnesota 2021
Based off the work done previously by Sydney Williams and Jeff Fessler at U of Michigan

# Arguments
-`x0` initial estimate

-`∇f` function returning gradient

-`Step` user-selected step size (1/Lipschitz)

-`g_prox` function performing proximal ste

-`Fcost` function calculating cost at each iteration


# Optional Arguments:
-`niter` total number of iterations

# Outputs:
-`xs`          [np niter]	    estimates each iteration
"""
function ir_fista(x0,Fcost::Function,∇f::Function,step::Real ;
    g_prox::Function=(z, step::Real) -> z,
    niter=50,
    T=ComplexF32,
)
    #= 
    Generic FISTA method with cost calculation
    Fast Iterative Shrinkage/Thresholding Algorithm [FISTA]
    Modified momentum to give POGM

    Adapted by Taylor Froelich; Minnesota 2021
    Based off the work done previously by Sydney Williams and Jeff Fessler at U of Michigan
 
    Input Arguments:
    x0          [np 1]         initial estimate
    ∇f          [Fun]          function returning gradient:
    Step		[Fun]	       user-selected step size (1/Lipschitz)
    g_prox      [Fun]          function performing proximal ste
    Fcost      [Fun]          function calculating cost at each iteration
 
    Optional Arguments:

    niter       [1]            total number of iterations

    Outputs:
 	xs          [np niter]	    estimates each iteration =#
    @views begin
        # xs = zeros(eltype(T),length(x0),Int(niter))
        cost = zeros(Float64, Int(niter))

        x = zeros(ComplexF32, size(x0))
        v = zeros(ComplexF32, size(x0))

        x_old = x0
        t_old = 1

        @inbounds for iter = 1:niter
        x .= g_prox(v - step .* ∇f(v), step)

        t = (1 + sqrt(1 + 4 * t_old^2)) / 2
        v .= x + (t_old - 1) / t * (x - x_old) + (t_old / t) * (x - v)

        x_old = x
        t_old = t
            
            # Calculates cost at each iteration
            # xs[:,iter] .= vec(x);
        cost[iter] = Fcost(x);               
    end
    end
    return vec(x), cost
end
export ir_pogm
function ir_pogm(x0,Fcost::Function,∇f::Function,step::Real;
    restart::Symbol=:gr, # :fr :none
    restart_cutoff::Real=0.,
    bsig::Real=1,
    niter::Int=10,
    g_prox::Function=(z, c::Real) -> z,
)

	# initialize parameters
	told = 1
    sig = 1
    zetaold = 1

	# initialize x
	xold = zeros(ComplexF32, size(x0))
	yold = zeros(ComplexF32, size(x0))
	uold = zeros(ComplexF32, size(x0))
	zold = zeros(ComplexF32, size(x0))
	Fcostold = Fcost(x0)
	Fgradold = zeros(ComplexF32, size(x0)) # dummy

    cost = zeros(Float64, Int(niter))

	xnew = zeros(ComplexF32, size(x0))
	ynew = zeros(ComplexF32, size(x0))

    # iterations
    for iter = 1:niter
        fgrad = ∇f(xold)

        is_restart = false

        # gradient update for POGM [see KF18]
        unew = xold - step * fgrad
        # restart + "gamma" decrease conditions checked later for POGM,
        # unlike PGM, FPGM above

        # momentum coefficient "beta"
        if iter == niter
            tnew = 0.5 * (1 + sqrt(1 + 8 * told^2))
        else
            tnew = 0.5 * (1 + sqrt(1 + 4 * told^2))
        end
        beta = (told - 1) / tnew

        # momentum coefficient "gamma"
        gamma = sig * told / tnew

        znew = (unew + beta * (unew - uold) + gamma * (unew - xold)
                - beta * step / zetaold * (xold - zold))
        zetanew = step * (1 + beta + gamma)
        xnew .= g_prox(znew, zetanew) # non-standard PG update for POGM

        # non-standard composite gradient mapping for POGM:
        Fgrad = fgrad - 1 / zetanew * (xnew - znew)
        ynew .= xold - step * Fgrad
        Fcostnew = Fcost(xnew)

        # restart + "gamma" decrease conditions for POGM
        if restart != :none
            # function/gradient restart
            if ((restart === :fr && Fcostnew > Fcostold)
            || (restart === :gr && gr_restart(Fgrad, ynew - yold, restart_cutoff)))
                tnew = 1
                sig = 1
                is_restart = true

            # gradient "gamma" decrease
            elseif sum(Float64, real(Fgrad .* Fgradold)) < 0
                sig = bsig * sig
            end

            Fcostold = Fcostnew
            Fgradold = Fgrad
        end

        uold = unew
        zold = znew
        zetaold = zetanew

        xold = xnew
        yold = ynew
        cost[iter] = Fcost(xnew)

    end # for iter
    
	return xnew, cost
end
export ir_fpgm
function ir_fpgm(x0,Fcost::Function,∇f::Function,step::Real ;
    restart::Symbol=:gr, # :fr :none
    restart_cutoff::Real=0.,
    niter::Int=10,
    g_prox::Function=(z, c::Real) -> z,
)
    @views begin
        # initialize parameters
        told = 1

        # initialize x
        xold = zeros(ComplexF32, size(x0))
        yold = zeros(ComplexF32, size(x0))
        Fcostold = Fcost(x0)

        cost = zeros(Float64, Int(niter))

        xnew = zeros(ComplexF32, size(x0))
        ynew = zeros(ComplexF32, size(x0))


        @inbounds for iter = 1:niter
        fgrad = ∇f(xold)

        is_restart = false

        ynew .= g_prox(xold - step * fgrad, step) # standard PG update
        Fgrad = -(1. / step) * (ynew - xold) # standard composite gradient mapping
        Fcostnew = Fcost(ynew)

            # restart condition
        if restart != :none
                # function/gradient restart
            if ((restart === :fr && Fcostnew > Fcostold)
                || (restart === :gr && gr_restart(Fgrad, ynew - yold, restart_cutoff)))
                told = 1
                is_restart = true
            end
            Fcostold = Fcostnew
        end

        tnew = 0.5 * (1 + sqrt(1 + 4 * told^2))

            # momentum update
        xnew .= ynew + (told - 1) / tnew * (ynew - yold)

        xold = xnew
        yold = ynew
        told = tnew

        cost[iter] = Fcost(ynew)
    end # for iter
    end

	return ynew, cost
end
# ------
# Optimization Algorithms GPU
export Calc_Lipshitz_GPU
function Calc_Lipshitz_GPU(Enc)
    Z = CUDA.rand(eltype(Enc),size(Enc,2))
    @inbounds for i in 1:50
        Z = Enc'*(Enc*vec(Z)) / norm(Enc'*(Enc*vec(Z)))
    end
    return 2 * norm(Enc'*(Enc*Z))
end
export ir_fista_GPU
function ir_fista_GPU(
        x0::CuArray{<:Any},
        Fcost::Function,
        ∇f::Function,
        step::Real;
        g_prox::Function=(z, step::Real) -> z,
        niter=50,
)
    @views begin
        x = CUDA.zeros(eltype(x0),size(x0))
        v = CUDA.zeros(ComplexF32,size(x0))

        x_old = x0
        t_old = 1

        @inbounds for iter = 1:niter
            x .= g_prox(v - step .* ∇f(v), step)

            t = (1 + sqrt(1 + 4 * t_old^2)) / 2
            v .= x + (t_old - 1) / t * (x - x_old) + (t_old / t) * (x - v)

            x_old = x
            t_old = t
        end
    end
    return vec(x), Fcost(x)
end
export ir_pogm_GPU
function ir_pogm_GPU(
        x0::CuArray{<:Any},
        Fcost::Function,
        ∇f::Function,
        step::Real;
        restart::Symbol=:gr, # :fr :none
        restart_cutoff::Real=0.,
        bsig::Real=1,
        niter::Int=10,
        g_prox::Function=(z, c::Real) -> z,
)
    @views begin
        # initialize parameters
        told = 1
        sig = 1
        zetaold = 1

        # initialize x
        xold = CUDA.zeros(ComplexF32,size(x0))
        xnew = CUDA.zeros(ComplexF32,size(x0))
        
        yold = CUDA.zeros(ComplexF32,size(x0))
        ynew = CUDA.zeros(ComplexF32,size(x0))
        
        uold = CUDA.zeros(ComplexF32,size(x0))
        zold = CUDA.zeros(ComplexF32,size(x0))

        cost = CUDA.zeros(Float32,size(x0))
        Fcostold = Fcost(x0)
        Fgradold = CUDA.zeros(ComplexF32,size(x0)) # dummy


        # iterations
        @inbounds for iter = 1:niter
            fgrad = ∇f(xold)

            is_restart = false

            # gradient update for POGM [see KF18]
            unew = xold - step * fgrad
            # restart + "gamma" decrease conditions checked later for POGM,
            # unlike PGM, FPGM above

            # momentum coefficient "beta"
            if iter == niter
                tnew = 0.5 * (1 + sqrt(1 + 8 * told^2))
            else
                tnew = 0.5 * (1 + sqrt(1 + 4 * told^2))
            end
            beta = (told - 1) / tnew

            # momentum coefficient "gamma"
            gamma = sig * told / tnew

            znew = (unew + beta * (unew - uold) + gamma * (unew - xold) - beta * step / zetaold * (xold - zold))
            zetanew = step * (1 + beta + gamma)
            xnew = g_prox(znew, zetanew) # non-standard PG update for POGM

            # non-standard composite gradient mapping for POGM:
            Fgrad = fgrad - 1 / zetanew * (xnew - znew)
            ynew .= xold - step * Fgrad
            Fcostnew = Fcost(xnew)

            # restart + "gamma" decrease conditions for POGM
            if restart != :none
                # function/gradient restart
                if ((restart === :fr && Fcostnew > Fcostold) || (restart === :gr && gr_restart(Fgrad, ynew - yold, restart_cutoff)))
                    tnew = 1
                    sig = 1
                    is_restart = true

                # gradient "gamma" decrease
                elseif sum(Float32, real(Fgrad .* Fgradold)) < 0
                    sig = bsig * sig
                end

                Fcostold = Fcostnew
                Fgradold = Fgrad
            end

            uold = unew
            zold = znew
            zetaold = zetanew

            xold = xnew
            yold = ynew
        end # for iter
    end    
	return xnew, Fcost(xnew)
end
export ir_fpgm_GPU
function ir_fpgm_GPU(
        x0::CuArray{<:Any},
        Fcost::Function,
        ∇f::Function,
        step::Real;
        restart::Symbol=:gr, # :fr :none
        restart_cutoff::Real=0.,
        niter::Int=10,
        g_prox::Function=(z, c::Real) -> z,
)
    @views begin
        # initialize parameters
        told = 1

        # initialize x
        xold = CUDA.zeros(ComplexF32,size(x0))
        yold = CUDA.zeros(ComplexF32,size(x0))

        xnew = CUDA.zeros(ComplexF32,size(x0))
        ynew = CUDA.zeros(ComplexF32,size(x0))
        
        Fcostold = Fcost(x0)

        @inbounds for iter = 1:niter
            fgrad = ∇f(xold)

            is_restart = false

            ynew .= g_prox(xold - step * fgrad, step) # standard PG update
            Fgrad = -(1. / step) * (ynew - xold) # standard composite gradient mapping
            Fcostnew = Fcost(ynew)

                # restart condition
            if restart != :none
                    # function/gradient restart
                if ((restart === :fr && Fcostnew > Fcostold)
                    || (restart === :gr && gr_restart(Fgrad, ynew - yold, restart_cutoff)))
                    told = 1
                    is_restart = true
                end
                Fcostold = Fcostnew
            end

            tnew = 0.5 * (1 + sqrt(1 + 4 * told^2))

                # momentum update
            xnew .= ynew + (told - 1) / tnew * (ynew - yold)

            xold = xnew
            yold = ynew
            told = tnew
        end # for iter
    end

	return ynew, Fcost(ynew)
end
# ------
# GPU Recons
export GPU_unReg_LS
"""
` x1, cost = GPU_unReg_LS( ; ...)`

# Arguments
- `A_CPU`
- `Vector_CPU`
- `nier_GPU`
- `method`
- `device_number`
"""
function GPU_unReg_LS(
        A_CPU,
        Vector_CPU;
        nier_GPU::Int=100,
        method::Symbol=:pogm,
        device_number::Int=3,
)
    device!(device_number)
    if has_cuda_gpu()
        CUDA.allowscalar(false)
    end

    if typeof(A_CPU) <: SparseMatrixCSC{eltype(A_CPU), Int}
        A_GPU = CuSparseMatrixCSR(A_CPU)
    else
        A_GPU = CuArray(A_CPU)
    end

    Vector_GPU = CuArray(Vector_CPU)
    x0_GPU = CUDA.zeros(eltype(A_CPU),size(A_CPU,2))

    stepsize = Calc_Lipshitz_GPU(A_GPU)^(-1)

    ∇f = (x) -> (A_GPU' * (A_GPU * x - vec(Vector_GPU)))
    cost = (x) -> (0.5) * norm(A_GPU * x - vec(Vector_GPU))^2
    
    if method === :pogm
        out, cost = ir_pogm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
    elseif method === :fist
        out,cost = ir_fista_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
    elseif method === :fpgm
        out, cost = ir_fpgm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
    end

    return Array(out), cost
end
export GPU_Reg_LS
"""
` x1, cost = GPU_Reg_LS( ; ...)`

# Arguments
- `A_CPU`
- `Vector_CPU`
- `reg`
- `nier_GPU`
- `method`
- `device_number`
"""
function GPU_Reg_LS(
        A_CPU, 
        Vector_CPU,
        reg::AbstractFloat; 
        nier_GPU::Int=100,
        method::Symbol=:pogm,
        device_number::Int = 3,
)
"""
Regularized Least Squares with an L1 norm prox 
` vector_out, cost = GPU_Reg_LS( ; ...)`

# Arguments
- `Enc::AbstractArray{<:Any}` Encoding Matrix
- `eng_levs::AbstractArray{<:Any}` Energy levels to extract. Of the format like [0.99.0.1]
- `T::DataType = Float32` Data type
"""
    device!(device_number)

    if has_cuda_gpu()
        CUDA.allowscalar(false)
    end

    if typeof(A_CPU) <: SparseMatrixCSC{eltype(A_CPU), Int}
        A_GPU = CuSparseMatrixCSR(A_CPU)
    else
        A_GPU = CuArray(A_CPU)
    end

    Vector_GPU = CuArray(Vector_CPU)
    x0_GPU = CUDA.zeros(eltype(A_CPU),size(A_CPU,2))

    stepsize = Calc_Lipshitz_GPU(A_GPU)^(-1)

    ∇f = (x) -> (A_GPU' * (A_GPU * x - Vector_GPU)) + reg * x
    cost = (x) -> (0.5) * norm(A_GPU * x - Vector_GPU)^2  + reg/2 * norm(x)^2

    soft = (x,c) -> sign(x) * max(abs(x) - c, 0) # soft thresholding
    prox = (x,c) -> soft.(x, reg * c) # proximal operator
    
    if method === :pogm
        out, cost = ir_pogm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU, g_prox = prox)
    elseif method === :fist
        out,cost = ir_fista_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU, g_prox = prox)
    elseif method === :fpgm
        out, cost = ir_fpgm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU, g_prox = prox)
    end

    return Array(out), cost
end
# ----
# Not working section of GPU Code
# Problem with calar Indexing
function GPU_Reg_LS_Edge(A_CPU, Vector_CPU, nobj; 
    nier_GPU=100,
    δ=0.1,
    β=0.01,
    device_number = 3,
)
    device!(device_number)
    if has_cuda_gpu()
        CUDA.allowscalar(false)
    end

    reg = β * size(A_CPU,2)
    T_arg = Finite_diff(nobj)

    #A_GPU = CuSparseMatrixCSR(A_CPU)
    A_GPU = CuArray(A_CPU)

    Vector_GPU = CuArray(Vector_CPU)
    x0_GPU = CUDA.zeros(ComplexF32,size(A_CPU,2))

    stepsize = Calc_Lipshitz_GPU(A_GPU)^(-1)

    pot = (z,del) -> del^2 * (abs(z)/del - log(1 + abs(z)/del)) # Fair potential function
    dpot = (z,del) -> z / (1 + abs(z)/del) # potential derivative
    ∇f = (x) -> A_GPU'*(A_GPU*x - vec(Vector_GPU)) + reg * (T_arg' * dpot.(T_arg * x, δ))
    cost = (x) -> 1/2 * norm(A_GPU * x - vec(Vector_GPU))^2 + reg * sum(pot.(T_arg * x, δ))

    # soft = (x,c) -> sign(x) * max(abs(x) - c, 0) # soft thresholding
    # prox = (x,c) -> soft.(x, reg * c) # proximal operator
    
    if method === :pogm
        out, cost = ir_pogm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
    elseif method === :fist
        out,cost = ir_fista_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
    elseif method === :fpgm
        out, cost = ir_fpgm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
    end

    return Array(out2), cost2
end
# Async running of GPUs Not working
function GPUs_Par(A_GPU, Vector_GPU,stepsize, nier_GPU, reg)
    @sync begin
        @async begin
            device!(2)
            ∇f = (x) -> (A_GPU' * (A_GPU * x - vec(Vector_GPU)))
            cost = (x) -> (0.5) * norm(A_GPU * x - vec(Vector_GPU))^2
            out1, ~ = ir_pogm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
        end
        @async begin
            device!(3)
            ∇f = (x) -> (A_GPU' * (A_GPU * x - Vector_GPU)) + reg * x
            cost = (x) -> (0.5) * norm(A_GPU * x - Vector_GPU)^2  + reg/2 * norm(x)^2
            Array(out2), ~ = ir_pogm_GPU(x0_GPU, cost, ∇f, stepsize; niter = nier_GPU)
        end
    end
    return Array(out1), Array(out2)
end
function Run_GPUS(A_CPU,Vector_CPU,nier_GPU,reg)
    A_GPU = CuArray(A_CPU)
    Vector_GPU = CuArray(Vector_CPU)
    x0_GPU = CUDA.zeros(ComplexF32,size(A_CPU,2))

    synchronize()

    stepsize = Calc_Lipshitz_GPU(A_GPU)^(-1)

    ou1, out2 = GPUs_Par(A_GPU, Vector_GPU, stepsize, nier_GPU, reg)

    return out1, out2
end
# ---
# Various Helper Functions
export ndgrid
function ndgrid(
		x::AbstractVector{<:Any},
		y::AbstractVector{<:Any}; T=Float32)

	tmp = Iterators.product(x, y)
	return [p[1] for p in tmp], [p[2] for p in tmp]
end
function ndgrid(
		x::AbstractVector{<:Any},
		y::AbstractVector{<:Any},
		z::AbstractVector{<:Any})

	tmp = Iterators.product(x, y, z)
	return [p[1] for p in tmp], [p[2] for p in tmp], [p[3] for p in tmp]
end
export nrmse_Images
function nrmse_Images(X_exp, X_true)
    return norm(X_exp - X_true) / norm(X_true)
end
export Finite_diff
function Finite_diff(nobj)
    Dn = spdiagm(-1 => -ones(nobj[2] - 1), 0 => 2 * ones(nobj[2]), 1 => -ones(nobj[2] - 1))[2:end - 1,:]
    Dm = spdiagm(-1 => -ones(nobj[1] - 1), 0 => 2 * ones(nobj[1]), 1 => -ones(nobj[1] - 1))[2:end - 1,:]

    out = vcat(kron(sparse(I(nobj[1])), Dn), kron(Dm, sparse(I(nobj[2]))))
end
# ---
# Sparsity Operator
export sparsify_operator
"""
`Enc_sparse = sparsify_operator( ; ...)`
Enc has Dimensions x_pts by y_pts by RO
Eng_levs is like [0.99, 0.1] etc

# Arguments
- `Enc::AbstractArray{<:Any}` Encoding Matrix
- `eng_levs::AbstractArray{<:Any}` Energy levels to extract. Of the format like [0.99.0.1]
- `T::DataType = Float32` Data type
"""
function sparsify_operator(
    Enc::AbstractArray,
    eng_levs::AbstractArray;
    T::DataType = Float32,
)
    Enc_type = eltype(Enc)
    num_levels = length(eng_levs)

    sz_M  = size(Enc)
    t = sz_M[end]
    num_pix = prod(sz_M[1:end-1])

    S = Array{SparseMatrixCSC{Enc_type, Int},1}(undef,num_levels)

    u_F = Array{Array{Int,1},2}(undef,t,num_levels)
    v_F = Array{Array{Int,1},2}(undef,t,num_levels)
    w_F = Array{Array{Enc_type,1},2}(undef,t,num_levels)

    @inbounds @views begin
            Threads.@threads for index_tim = 1:t
                # Extract each frame

                frm = Enc[:,:,index_tim]
                #frm = fftshift(ifft(Enc[:,:,index_tim]))
                # Extract the locations & values that constitute
                # Elev fraction of the total energy in the frame

                sm = sum(sum(abs.(frm).^2)); #total energy

                srt = sort(abs.(vec(frm)), rev=true)
                inx = sortperm(abs.(vec(frm)),rev = true)
                cs = cumsum(srt.^2)

                for idx = 1:num_levels
                    Elev = eng_levs[idx]
                    off = 0
                    fnum = findall(x -> x >= Elev,cs./sm)

                    if !isempty(fnum)
                        numkeep = fnum[1]
                    else
                        numkeep = prod(size(frm))
                    end       

                    u_F[index_tim,idx] = index_tim.*ones(numkeep-off)
                    v_F[index_tim,idx] = inx[1+off:numkeep]
                    w_F[index_tim,idx] = eltype(Enc_type).(frm[inx[1+off:numkeep]])
                end
            end

        for idx = 1:num_levels
            uF = reduce(vcat, vec(u_F[:,idx]))
            vF = reduce(vcat, vec(v_F[:,idx]))
            wF = reduce(vcat, vec(w_F[:,idx]))

            S[idx] = sparse(eltype(Int).(uF), eltype(Int).(vF), wF, t, num_pix)
        end
    end
    return S
end