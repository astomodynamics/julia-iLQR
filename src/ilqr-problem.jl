################################################################################
#=
    ilqr solver environment
=#
################################################################################

export iLQRProblem

# DDP solver as a Julia class
mutable struct iLQRProblem
    dt::Float64 # discretization step-size
    tN::Int64 # number of time-discretization steps
    max_ite::Int64

    x_dim::Int64 # state dimension
    u_dim::Int64 # control dimension

    X::AbstractArray{Float64,2} # X trajectory storage
    U::AbstractArray{Float64,2} # U trajectory storage

    X_temp::AbstractArray{Float64,2} # temporal X trajectory storage
    U_temp::AbstractArray{Float64,2} # temporal U trajectory storage

    J::Float64 # cost

    k_arr::AbstractArray{Float64,2}
    K_arr::AbstractArray{Float64,3}

    reg_param1::Float64 # regulation parameter #1
    reg_param2::Float64 # regulation parameter #2
    reg_param1_fact::Float64 # regulation factor #1
    reg_param2_fact::Float64 # regulation factor #2

    line_search_steps::AbstractArray{Float64,1}  # learning rate

    sim_new_traj::Bool
end

# The problem is set in the class constructor
function iLQRProblem(
    model::Any;
    islinear=model.islinear,
    isconstrained=model.isconstrained,
    istochastic=model.isstochastic
)
    dt = model.dt
    tN = model.tN
    max_ite = model.max_ite

    x_dim, u_dim = model.x_dim, model.u_dim

    const_funcs = get_constraint_funcs(model)
    c = const_funcs.constraints(model, zeros(x_dim), zeros(u_dim))
    λ_dim = size(c, 1)
    model.λ_dim = λ_dim

    J = 0

    reg_param1 = 1e-0 # now best 1e-10, 1e-9,
    reg_param2 = 1e-0 # 1*1e1, 8e0
    reg_param1_fact = 5
    reg_param2_fact = 10
    μip = 0

    line_search_steps = 4 .^ LinRange(0, -5, 15)
    # line_search_steps = 2 .^ LinRange(0, -10, 11)

    sim_new_traj = false
    isfeasible = false

    iLQRProblem(
        dt,
        tN,
        max_ite,
        x_dim,
        u_dim,
        Array{Float64}(undef, x_dim, tN),
        Array{Float64}(undef, u_dim, tN),
        Array{Float64}(undef, x_dim, tN),
        Array{Float64}(undef, u_dim, tN),
        J,
        Array{Float64}(undef, u_dim, tN),
        Array{Float64}(undef, u_dim, x_dim, tN),
        reg_param1,
        reg_param2,
        reg_param1_fact,
        reg_param2_fact,
        line_search_steps,
        sim_new_traj,
    )
end


"""
    reset_ddp_problems()
"""
function reset_ddp_problems(
    model::Any,
    problem::iLQRProblem,
    sim_new_traj::Bool,
    islinear::Bool,
    isconstrained::Bool,
    isstochastic::Bool,
)
    x_dim, u_dim, λ_dim = problem.x_dim, problem.u_dim, problem.λ_dim
    tN = problem.tN

    if sim_new_traj
        X, U = initialize_trajectory(model, tN)
        
        problem.X, problem.U = copy(X), copy(U)
        problem.sim_new_traj = false

        #    if ddpaccel
        #         solve_ddp()
        #    end
    end
    problem.islinear = islinear
    problem.isstochastic = isstochastic
    isfeasible = get_feasibility(model, problem.X, problem.U)
    problem.isfeasible = isfeasible

    if !isconstrained
        problem.Λ = zeros(λ_dim, tN)
        problem.Y = zeros(λ_dim, tN)
    elseif isconstrained && isfeasible
        # Λ = 1.0 * ones(λ_dim, tN)
        # problem.Λ = 1.0 * ones(λ_dim, tN)
        problem.Λ = 0.0 * ones(λ_dim, tN)
    elseif isconstrained && !isfeasible
        problem.Λ = 100.0 * ones(λ_dim, tN)
        problem.Y = 10.0 * ones(λ_dim, tN)
    end

    problem.J = trajectory_cost(model, problem.X, problem.U)

    if isconstrained && problem.μip == 0
        # problem.μip = problem.J / tN 
        problem.μip = problem.J / tN / λ_dim 
    end

    problem.k_arr = zeros(u_dim, tN) # feedforward gain
    problem.K_arr = zeros(u_dim, x_dim, tN) # feedback gain
    problem.h_arr = zeros(λ_dim, tN) # feedforward gain for constraints
    problem.H_arr = zeros(λ_dim, x_dim, tN) # feedback gain for constraints
    problem.p_arr = zeros(λ_dim, tN)
    problem.P_arr = zeros(λ_dim, x_dim, tN)
end

