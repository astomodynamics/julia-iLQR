
#= 
    helper funcitons
=#

export 
    initialize_trajectory,
    rk4_step,
    simulate_trajectory

"""
    initialize_trajectory(model, tN)

Initialize the state and control trajectory

# Arguments
- `model`: AbstractModel
- `tN`: trajectory time range

# Return
- X
- U
"""
function initialize_trajectory(
    model::Any, 
    tN::Int64; 
    x_init=model.x_init, 
    dt=model.dt,
    x_dim=model.x_dim,
    u_dim=model.u_dim,
    dynamics=model.dynamics,
)
    
    X = zeros(x_dim, tN)
    U = zeros(u_dim, tN - 1)
    # U = 0.02 * rand(Float64,(u_dim, tN-1)) .- 0.01
    X[:, 1] = copy(x_init)

    # Run simulation with substeps
    for t in axes(U,2)
        X[:, t+1] =
                X[:, t] + rk4_step(model, dynamics, t, X[:, t], U[:, t], dt) * dt
    end
    return X, U
end

"""


simulate dynamics given initial condition and control sequence

# Arguments

"""
function simulate_trajectory(
    model::Any,
    x0::AbstractArray{Float64,1},
    U::AbstractArray{Float64,2},
    dt::Float64=model.dt;
    isstochastic=false,
    dynamics=model.dynamics,
)   
    tN = size(U,2)+1
    X = zeros(size(x0,1),tN)
    X[:, 1] = copy(x0)

    for t in axes(U,2)
        if !isstochastic
            X[:, t+1] =
                X[:, t] + rk4_step(model, dynamics, t, X[:, t], U[:, t], dt) * dt
        end
    end

    return X
end


"""
    rk4(model, f, t, x, u, h)

Returns one step of runge-kutta ode step with fixed time length

# Arguments
- `model` (Model):
- `f` (Function):
- `t`:
- `x`:
- `u`:
- `h`: discrete time step

# Returns
- `ẋ`:

"""
function rk4_step(
    model::Any,
    f::Function,
    t::Int64,
    x::AbstractArray{Float64,1},
    u::AbstractArray{Float64,1},
    h::Float64,
)

    k1 = f(model, x, u, t+0.0, t)
    k2 = f(model, x + h / 2.0 * k1, u, t + h / 2.0, t)
    k3 = f(model, x + h / 2.0 * k2, u, t + h / 2.0, t)
    k4 = f(model, x + h * k3, u, t + h, t)

    return (k1 + 2 * k2 + 2 * k3 + k4)/6
end
