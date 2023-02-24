#=
###############################################################################
#   iterative Linear Quadratic  Algorithm is implemented
###############################################################################
=#



"""
    solve_ilqr(model, problem)

The deterministic optimal control problem is solved by
differential dynamic programming

# Arguments
- `model`: dynamics model constructor
- `problem`: ilqr problem constructor
"""
function solve_ilqr(
    model::Any,
    problem::iLQRProblem,
    tN,
    max_ite;
    print::Bool=true
)
    println("> Solve iLQR Problem \n")


    reset_ilqr_problem(model, problem, true)
    J = copy(problem.J)

    J_old = J
    success, ite = false, 0
    while (ite < max_ite) &&
        !(success && (abs(J_old - J) / J) < model.conv_tol)
        if print
            if (mod(ite, 5) == 0)
                println("------------------------------------------------\n\
                         iter   || cost   \n\
                         ------------------------------------------------\n")
            end
            println(
                "$(ite+1)  ||  $J \n",
            )
        end

        backward_pass(model, problem)
        forward_pass(model, problem)

        J_new = problem.J
        if J_new < J_old
            success = true
            J_old = copy(J)
            J = copy(J_new)
        end
        ite += 1
    end
    println("------------------------------------------------\n\
        >>> iLQR Iterations Finished <<<\n\
        total iterations: ",
        ite,
        "\n",
        "cost: ",
        J,
        "\n",
        "cost threshold: ",
        abs(J_old - J) / J,
        "\n--------------------------------\n",
    )
    return problem.X, problem.U
end

function backward_pass(model::Any, problem::iLQRProblem)
    X, U = problem.X, problem.U
    x_dim, u_dim = model.x_dim, model.u_dim
    reg_param1, reg_param2 = problem.reg_param1, problem.reg_param2
    tN = model.tN
    dt = model.dt

    # initialize dynamics, cost functions, and action-value functions
    fx_arr = zeros(x_dim, x_dim, tN)
    fu_arr = zeros(x_dim, u_dim, tN)

    l_arr = zeros(tN)
    lx_arr = zeros(x_dim, tN)
    lxx_arr = zeros(x_dim, x_dim, tN)
    lu_arr = zeros(u_dim, tN)
    luu_arr = zeros(u_dim, u_dim, tN)
    lxu_arr = zeros(x_dim, u_dim, tN)

    # Storing dynamics and costs
    for t in 1:tN-1
        fx, fu = get_jacobian(model, X[:, t], U[:, t], t)
        fx_arr[:, :, t] = I + fx * dt
        fu_arr[:, :, t] = fu * dt

        # compute running cost at each time step
        (l, lx, lxx, lu, luu, lxu) = instantaneous_cost(model, X[:, t], U[:, t], gradient=true)
        l_arr[t] = l 
        lx_arr[:, t] = lx 
        lxx_arr[:, :, t] = lxx
        lu_arr[:, t] = lu 
        luu_arr[:, :, t] = luu 
        lxu_arr[:, :, t] = lxu 
    end

    # push the terminal cost
    l_arr[end], lx_arr[:, end], lxx_arr[:, :, end] =
        terminal_cost(model, X[:, end], gradient=true)

    # initialize value function and the derivatives
    V = copy(l_arr[end])
    Vx = copy(lx_arr[:, end])
    Vxx = copy(lxx_arr[:, :, end])

    # initialize feedforward and feedback gains
    k_arr = zeros(u_dim, tN) # feedforward gain
    K_arr = zeros(u_dim, x_dim, tN) # feedback gain

    # backward rollout
    for t in tN-1:-1:1
        fx, fu = fx_arr[:, :, t], fu_arr[:, :, t]
        lx, lu, lxx, lxu, luu = lx_arr[:, t], lu_arr[:, t], lxx_arr[:, :, t], lxu_arr[:, :, t], luu_arr[:, :, t]

        # Q = l + V
        Qx = lx + fx' * Vx
        Qu = lu + fu' * Vx
        Qxx = lxx + fx' * (Vxx + reg_param1 * I) * fx
        Qxu = lxu + fx' * (Vxx + reg_param1 * I) * fu
        Quu = luu + fu' * (Vxx + reg_param1 * I) * fu

        # second regulation
        Quu_eig = eigen(Quu)
        if minimum(real(Quu_eig.values)) <= 0.0
            Quu += reg_param2 * I
            println("Quu is not positive definite")
        end

        kK = -Quu \ [Qu Qxu']
        k = kK[:, 1]
        K = kK[:, 2:end]

        Vx = Qx + K' * Qu + K' * Quu * k + Qxu * k
        Vxx = Qxx + K' * Qxu' + Qxu * K + K' * Quu * K

        k_arr[:, t] = copy(k)
        K_arr[:, :, t] = copy(K)
    end

    problem.k_arr, problem.K_arr = copy(k_arr), copy(K_arr)

    return nothing
end

function forward_pass(model::Any, problem::iLQRProblem)
    dt = model.dt
    k_arr, K_arr = problem.k_arr, problem.K_arr
    X_old, U_old = problem.X, problem.U

    X_new = zeros(size(X_old))
    U_new = zeros(size(U_old))

    X_new[:, 1] = copy(X_old[:, 1])
    J_old = copy(problem.J)
    J_new = []

    line_search_steps = problem.line_search_steps
    dynamics = model.dynamics

    for (ind, step) in enumerate(line_search_steps)
        for t in axes(U_old, 2)
            k = k_arr[:, t]
            K = K_arr[:, :, t]
            U_new[:, t] = U_old[:, t] +
                          step * k + K * (X_new[:, t] - X_old[:, t])

            X_new[:, t+1] = X_new[:, t] +
                            rk4_step(model, dynamics, t, X_new[:, t], U_new[:, t], dt) * dt

        end

        # evaluate the new trajectory
        J_new = trajectory_cost(model, X_new, U_new)

        if J_new < J_old
            problem.X, problem.U, problem.J = copy(X_new), copy(U_new), copy(J_new)
            problem.reg_param1 /= problem.reg_param1_fact 
            problem.reg_param2 /= problem.reg_param2_fact

            return nothing
        end
    end

    if J_new >= J_old
        problem.X, problem.U, problem.J = X_new, U_new, J_new
        problem.reg_param1 *= problem.reg_param1_fact
        problem.reg_param2 *= problem.reg_param2_fact

        return nothing
    end
    return nothing
end