#=
    cost definitions
=#

# Dynamics model
abstract type AbstractModel end

export trajectory_cost,
       instantaneous_cost,
       terminal_cost


"""
    trajectory_cost()

"""
function trajectory_cost(model::AbstractModel, X::AbstractArray{Float64,2}, U::AbstractArray{Float64,2})
    J = 0
    dt = model.dt
    for t in axes(U,2)
        # Jt = instantaneous_cost(model, X[:, t], U[:, t])
        Jt = state_cost(model, X[:, t]) + control_cost(model, U[:,t])
        J += Jt * dt
    end
    Jt = terminal_cost(model, X[:, end])
    J += Jt
    return J
end

"""
    state_cost(model, x)

Compute state cost at a specific time 

# Arguments
- `model`: Abstract model
- `x`: state

# Return
- `q(x)`: state cost
"""
function state_cost(
    model::AbstractModel, 
    x::AbstractArray{Float64,1},
)
    q = x[1]^2 + 1/2 * x[2]^2 + 50 * (1 + cos(x[3]))^2 +  1/2 * x[4]^2
    return q
end

"""
    control_cost(model, u)

Compute control cost at a specific time 

# Arguments
- `model`: Abstract model
- `u`: control 

# Return
- `qᵤ(u)`: control cost
"""
function control_cost(
    model::AbstractModel, 
    u::AbstractArray{Float64,1},
)
    R = model.R
    return 1 / 2 *transpose(u) * R * u
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
        # ϕ = 1 / 2 * transpose(x - x_final) * F * (x - x_final)
        ϕ = 0
        return ϕ    
    else
        l = 1 / 2 * transpose(x - x_final) * F * (x - x_final)
        ∇ₓϕ = F * transpose(x - x_final)
        ∇ₓₓϕ = F
        return ϕ, ∇ₓϕ, ∇ₓₓϕ
    end
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
    model::AbstractModel, 
    x::AbstractArray{Float64,1}, 
    u::AbstractArray{Float64,1}; 
    gradient::Bool=false
)
    Q = model.Q
    R = model.R
    
    if !gradient
        ℓ = 1 / 2 * transpose(x) * Q * x + 1 / 2 *transpose(u) * R * u
        # ℓ = x[1]^2 + x[2]^2 + 500*(1 + cos(x[3]))^2 + x[4]^2
        return ℓ
    else
        ℓ = 1 / 2 * transpose(x) * Q * x + 1 / 2 *transpose(u) * R * u
        ∇ᵤℓ = R * u
        ∇ᵤᵤℓ = R
        ∇ₓℓ = Q * x
        ∇ₓₓℓ = Q
        return ℓ, ∇ᵤℓ, ∇ᵤᵤℓ, ∇ₓℓ, ∇ₓₓℓ
    end
end