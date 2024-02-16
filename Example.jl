cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.precompile()

using Statistics, StaticArrays, SparseArrays
using LinearAlgebra, DSP, FFTW
using Plots
using MIRT: ir_mri_sensemap_sim, ellipse_im
using Images: imresize
using HDF5
using CUDA, CUDA.CUSPARSE
using Wavelets
device!(3)

## --- Recon
iter_global = 200

# With sparsity
S = @time sparsify_operator(permutedims(reshape(Mxy_total,(prod(size(Mxy_total)[1:2]),nobj[1],nobj[2])),[2,3,1]),[1])
Enc, F, Signal_new, ~, Object_original = @time Signal_Enc(permutedims(S[1],[2,1]),nobj)

# Without sparsity
#Enc, F, Signal_new, ~, Object_original = @time Signal_Enc(permutedims(reshape(Mxy_total,(prod(size(Mxy_total)[1:2]),nobj[1],nobj[2])),[2,3,1]), nobj)
# ----
Signal_tmp = reshape(reshape(reshape(Object_original,1,:) * Enc',:)',size(Mxy_total,1),size(Mxy_total,2))
Signal_tmp = Signal_tmp .+ (0 * randn(Complex{eltype(T)},size(Signal_tmp)))
recon_tmp = reverse(ifftshift(fft(Signal_tmp)),dims=2)

stepsize = Calc_Lipshitz(F)^(-1)
## ---
# UnReg Least Squares Problem

∇f = (x) -> F'*(F*x - vec(Signal_tmp))
cost = (x) -> 1/2 * norm(F * x - vec(Signal_tmp))^2
x0 = zeros(eltype(Complex{T}),prod(nobj))

xs1,cost1 = @time ir_pogm(x0, cost, ∇f, stepsize; niter = iter_global)
Final_image1 = reverse(reverse(abs.(reshape(xs1[:,end],(nobj[1],nobj[2])))',dims=1),dims=2)

## ----------------------------------------------------------------------
# Tikhonov regularization
#=
The Tikhonov regularizer R(x) = 1/2 ∥x∥2 discourages large values of x
̌x̌ = argmin Ψ(𝐱)
Ψ(𝐱) ≜ 1/2*(||𝐀𝐱-𝐲||₂)² + 1/2*β*(||𝐱||₂)²

The cost function:
1/2*(||𝐀𝐱-𝐲||₂)² + β*R(𝐱)
The regularizer:
R(𝐱) = ψ(𝐱)

ψ is a quadratic potential function
ψ(z) = |z|²/2

Julia Code:
pot = (z) -> abs(z)^2/2         # Quadratic potential function
cost = (x) -> 1/2 * norm(A * x - y)^2 + 1/2 * reg * sum(pot.(x))
dpot = (z) -> z                 # potential derivative
∇f = (x) -> A'*(A*x - y) + reg * dpot.(x)
=#
reg = @time (0.12 * opnorm(F)^2)
pot = (z) -> abs(z)^2/2
dpot = (z) -> z  # potential derivative
∇f = (x) -> F'*(F*x - vec(Signal_tmp)) + reg * dpot.(x)
cost = (x) -> 1/2 * norm(F * x - vec(Signal_tmp))^2 + reg * sum(pot.(x))
x0 = zeros(eltype(Complex{T}),prod(nobj))

xs2,cost2 = @time ir_pogm(x0, cost, ∇f, stepsize; niter = iter_global)
Final_image2 = reverse(reverse(abs.(reshape(xs2[:,end],(nobj[1],nobj[2])))',dims=1),dims=2)

## ----------------------------------------------------------------------
# Tikhonov regularization with Finite differences
# Need to Pay attention to Reg
#=
The Tikhonov regularizer R(x) = 1/2 ∥x∥2 discourages large values of x
Finite Differences discourages differences between neighbors.
̌x̌ = argmin Ψ(𝐱)
Ψ(𝐱) ≜ 1/2*(||𝐀𝐱-𝐲||₂)² + 1/2*β*(||𝐓𝐱||₂)²

The cost function:
1/2*(||𝐀𝐱-𝐲||₂)² + β*R(𝐱)
The regularizer:
R(𝐱) = ψ(𝐓𝐱)

ψ is a quadratic potential function
ψ(z) = |z|²/2
𝐓 is first-order finite differences in 2D,

Julia Code:
T = finite differences sparsifying transform for anisotropic TV
pot = (z) -> abs(z)^2/2         # Quadratic potential function
cost = (x) -> 1/2 * norm(A * x - y)^2 + 1/2 * reg * pot.(T*x)
dpot = (z) -> z                 # potential derivative
∇f = (x) -> A'*(A*x - y) + reg * (T' * dpot.(T * x))
=#
reg = 0.01*prod(nobj)
T_arg = Finite_diff(nobj)
pot = (z)-> abs(z)^2/2
dpot = (z) -> z # potential derivative
∇f = (x) -> F'*(F*x - vec(Signal_tmp)) + reg * (T_arg' * dpot.(T_arg * x))
cost = (x) -> 1/2 * norm(F * x - vec(Signal_tmp))^2 + reg * sum(pot.(T_arg * x))
x0 = zeros(eltype(Complex{T}),prod(nobj))

xs3,cost3 = @time ir_pogm(x0, cost, ∇f, stepsize; niter = iter_global)
Final_image3 = reverse(reverse(abs.(reshape(xs3[:,end],(nobj[1],nobj[2])))',dims=1),dims=2)
## ----------------------------------------------------------------------
# Elastic-net regularizer
#=
̌x̌ = argmin Ψ(𝐱)
Ψ(𝐱) ≜ 1/2*(||𝐀𝐱-𝐲||₂)² + 1/2*α*(||𝐱||₂)² + β*||𝐱||₁ 

The cost function:
1/2*(||𝐀𝐱-𝐲||₂)² + 1/2*α*(||𝐱||₂)² + β*||𝐱||₁ 
The gradient function:
(𝐀'* (𝐀𝐱-y)) + αx 

with L1 prox step

Julia Code:
∇f = (x) -> (F' * (F * x - vec(Signal_tmp))) + reg_paper * x
cost = (x) -> (0.5) * norm(F * x - vec(Signal_tmp))^2  + reg_paper/2 * norm(x)^2 + reg_paper*norm(x,1) 

soft = (x,c) -> sign(x) * max(abs(x) - c, 0) # soft thresholding
prox = (x,c) -> soft.(x, reg_paper * c) # proximal operator
=#
# ----------------------------------------------------------------------
α_reg = 1e5 #(0.15 * opnorm(F)^2);
β_reg = 1e4 #(0.15 * opnorm(F)^2);
∇f = (x) -> (F' * (F * x - vec(Signal_tmp))) + α_reg * x
cost = (x) -> (0.5) * norm(F * x - vec(Signal_tmp))^2  + α_reg/2 * norm(x)^2 + β_reg*norm(x,1) 

soft = (x,c) -> sign(x) * max(abs(x) - c, 0) # soft thresholding
prox = (x,c) -> soft.(x, β_reg * c) # proximal operator

x0 = zeros(eltype(Complex{T}),prod(nobj))

xs4,cost4 = @time ir_pogm(x0, cost, ∇f, stepsize; niter = iter_global,g_prox = prox)
Final_image4 = reverse(reverse(abs.(reshape(xs4[:,end],(nobj[1],nobj[2])))',dims=1),dims=2)
## ----------------------------------------------------------------------
# Edge-preserving regularization
#=
̌x̌ = argmin Ψ(𝐱)
Ψ(𝐱) ≜ 1/2*(||𝐀𝐱-𝐲||₂)² + β*R(𝐱)

The cost function:
1/2*(||𝐀𝐱-𝐲||₂)² + β*R(𝐱)
The regularizer:
R(𝐱) = ∑ψ(𝐓𝐱)

ψ is the Fair potential function
ψ(z) = δ²(|z|/δ - log(1 + |z|/δ)) 
𝐓 is first-order finite differences in 2D

Julia Code:
T = finite differences sparsifying transform for anisotropic TV
pot = (z,del) -> del^2 * (abs(z)/del - log(1 + abs(z)/del)) # Fair potential function
cost = (x) -> 1/2 * norm(A * x - y)^2 + reg * sum(pot.(T*x, delta))
dpot = (z,del) -> z / (1 + abs(z)/del) # potential derivative
∇f = (x) -> A'*(A*x - y) + reg * (T' * dpot.(T * x, delta))
=#
# ----------------------------------------------------------------------
reg = 5e3
delta = 0.1
T_arg = Finite_diff(nobj)
pot = (z,del) -> del^2 * (abs(z)/del - log(1 + abs(z)/del)) # Fair potential function
dpot = (z,del) -> z / (1 + abs(z)/del) # potential derivative
∇f = (x) -> F'*(F*x - vec(Signal_tmp)) + reg * (T_arg' * dpot.(T_arg * x, delta))
cost = (x) -> 1/2 * norm(F * x - vec(Signal_tmp))^2 + reg * sum(pot.(T_arg * x, delta))
x0 = zeros(eltype(Complex{T}),prod(nobj))

xs5,cost5 = @time ir_pogm(x0, cost, ∇f, stepsize; niter = iter_global)
Final_image5 = reverse(reverse(abs.(reshape(xs5[:,end],(nobj[1],nobj[2])))',dims=1),dims=2)
## --------------------------------------------------------------------------
# Wavelet sparsity in synthesis form
#=
taking advantage of a proximal operator

𝐖 is an orthogonal discrete (Haar) wavelet transform,
Because 𝐖 is unitary, we make the change of variables
𝐳 = 𝐖𝐱,
solve for 𝐳,
then let 𝐱 = 𝐖'𝐳 at the end.

̌x̌ = argmin Ψ(𝐱)
Ψ(𝐱) ≜ 1/2*(||𝐀𝐱-𝐲||₂)² + β*||𝐖𝐱||₁

The cost function:
1/2*(||𝐀𝐱-𝐲||₂)² + β*||𝐖𝐱||₁
The regularizer:
R(𝐱) = ||𝐖𝐱||₁

Julia Code:
regx =  0.03 * prod(nobj)
reg =   0.01 * prod(nobj)

cost = (x) -> 1/2 * norm(F * x - y)^2 + reg * norm(dwt(x,wavelet(WT.haar),3),1) # 1-norm regularizer
∇f = (x) -> F' * (F * x - y)

soft = (x,c) -> sign(x) * max(abs(x) - c, 0) # soft thresholding
prox = (x,c) -> idwt(soft.(dwt(x,wavelet(WT.haar),3), reg * c),wavelet(WT.haar),3) # proximal operator

=#
reg =   0.01 * prod(nobj)

cost = (x) -> 1/2 * norm(F * x - vec(Signal_tmp))^2 + reg * norm(dwt(x,wavelet(WT.haar),3),1) # 1-norm regularizer
∇f = (x) -> F' * (F * x - vec(Signal_tmp))

soft = (x,c) -> sign(x) * max(abs(x) - c, 0) # soft thresholding
prox = (x,c) -> idwt(soft.(dwt(x,wavelet(WT.haar),3), reg * c),wavelet(WT.haar),3) # proximal operator
x0 = zeros(eltype(Complex{T}),prod(nobj))

xs6,cost6 = @time ir_pogm(x0, cost, ∇f, stepsize; niter = iter_global,g_prox = prox)
Final_image6 = reverse(reverse(abs.(reshape(xs6[:,end],(nobj[1],nobj[2])))',dims=1),dims=2)
## ----------------------------------------------------------------------