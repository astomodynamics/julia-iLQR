#=
    cost definitions
=#

export trajectory_cost,
       instantaneous_cost,
       terminal_cost


"""
    trajectory_cost(model, X, U)


"""
function trajectory_cost(model::Any, X::AbstractArray{Float64,2}, U::AbstractArray{Float64,2})
    J = 0
    dt = model.dt
    for t in axes(U,2)
        Jt = instantaneous_cost(model, X[:,t], U[:,t])
        J += Jt * dt
    end
    Jt = terminal_cost(model, X[:, end])
    J += Jt
    return J
end

"""
    instantaneous_cost()

Compute the instantaneous cost at a specific time

# Arguments
- `model`: Abstract model
- `x`: state at a specific time step
- `u`: control at a specific time step
- `gradient`: first- and second-order derivative of terminal cost if needed

# Return
- `ℓ`: instantaneous cost at a specific time step
"""
function instantaneous_cost(
    model::Any, 
    x::AbstractArray{Float64,1}, 
    u::AbstractArray{Float64,1}; 
    gradient::Bool=false
)
    x_dim = model.x_dim
    u_dim = model.u_dim
    Q = model.Q
    R = model.R
    
    if !gradient
        ℓ = 1 / 2 * transpose(x) * Q * x + 1 / 2 *transpose(u) * R * u
        return ℓ
    else
        ℓ = 1 / 2 * transpose(x) * Q * x + 1 / 2 *transpose(u) * R * u
        ∇ₓℓ = Q * x
        ∇ₓₓℓ = Q
        ∇ᵤℓ = R * u
        ∇ᵤᵤℓ = R
        ∇ₓᵤℓ = zeros(x_dim, u_dim)
        return ℓ, ∇ₓℓ, ∇ₓₓℓ, ∇ᵤℓ, ∇ᵤᵤℓ, ∇ₓᵤℓ 
    end
end

"""
    terminal_cost(model, x, gradient)

Compute the terminal cost at the terminal time

# Arguments
- `model`: Abstract model
- `x`: state at the terminal step
- `gradient`: first- and second-order derivative of terminal cost if needed

# Return
- `ϕ`: terminal cost
"""
function terminal_cost(
    model::Any, 
    x::AbstractArray{Float64,1}; 
    gradient::Bool=false
)
    x_final = model.x_final
    F = model.F

    if !gradient
        ϕ = 1 / 2 * transpose(x - x_final) * F * (x - x_final)
        return ϕ    
    else
        ϕ = 1 / 2 * transpose(x - x_final) * F * (x - x_final)
        ∇ₓϕ = F * (x - x_final)
        ∇ₓₓϕ = F
        return ϕ, ∇ₓϕ, ∇ₓₓϕ
    end
end