#=
###############################################################################
#   iterative Linear Quadratic  Algorithm is implemented
###############################################################################
Refs:
    [1] Williams, G., Aldrich, A., and Theodorou, E. A.
    “Model Predictive Path Integral Control: From Theory to Parallel Computation.”
    Journal of Guidance, Control, and Dynamics, Vol. 40, No. 2, 2017, pp. 1–14.
    https://doi.org/10.2514/1.g001921.
=#


# Dynamics model
abstract type AbstractModel end

"""
        solve_ilqr_problems(model, cddp_problem)


# Arguments
- `model`
- `problem`
- `tN`

# Return

"""
function solve_ilqr_problems(
    model::Any,
    problem::DDPProblem;
    tN::Int64=problem.tN,
    max_ite::Int64=problem.max_ite,
    print::Bool=true
)
    println("------------------------------------------------\n\
            >>> Start DDP solver \n")
    println("Conditions:\n",
        "linear problem? = ", islinear, "\n",
        "constraints? = ", isconstrained, "\n",
        "stochasicity? = ", isstochastic, "\n",
    )

    sim_new_traj = true
    reset_ddp_problems(model, problem, sim_new_traj, islinear, isconstrained, isstochastic)

    if !isconstrained && !isstochastic
        solve_ddp(model, problem, tN, max_ite, print=print)
        println("")
    elseif isconstrained && !isstochastic
        solve_cddp_(model, problem, tN, max_ite, print=print)
        println("")
    elseif !isconstrained && isstochastic
        println("")
    elseif isconstrained && isstochastic
        println("")
    end

    return
end

"""
    solve_ilqr(model, problem)

The deterministic optimal control problem is solved by
the differential dynamic programming

# Arguments
- `model`
- `problem`
"""
function solve_ilqr(
    model::Any,
    problem::DDPProblem,
    tN,
    max_ite;
    print::Bool=true
)
    function _backward_pass(model::Any, problem::DDPProblem)
        X, U = problem.X, problem.U
        x_dim, u_dim = model.x_dim, model.u_dim
        reg_param1, reg_param2 = problem.reg_param1, problem.reg_param2
        tN = model.tN
        dt = model.dt

        # initialize dynamics, cost functions, and action-value functions
        fx_arr = zeros(x_dim, x_dim, tN)
        fu_arr = zeros(x_dim, u_dim, tN)
        fxx_arr = zeros(x_dim, x_dim, x_dim, tN)
        fxu_arr = zeros(x_dim, x_dim, u_dim, tN)
        fuu_arr = zeros(x_dim, u_dim, u_dim, tN)
        gx_arr = zeros(x_dim, x_dim, tN)
        gu_arr = zeros(x_dim, u_dim, tN)
        gxx_arr = zeros(x_dim, x_dim, x_dim, tN)
        gxu_arr = zeros(x_dim, x_dim, u_dim, tN)
        guu_arr = zeros(x_dim, u_dim, u_dim, tN)
        l_arr = zeros(tN)
        lx_arr = zeros(x_dim, tN)
        lxx_arr = zeros(x_dim, x_dim, tN)
        lu_arr = zeros(u_dim, tN)
        luu_arr = zeros(u_dim, u_dim, tN)
        lxu_arr = zeros(x_dim, u_dim, tN)

        # Storing dynamics and costs
        for t in 1:tN-1
            # compute quadratically approximated dynamics to get discrete dynamics
            # x(t+1) = F(x(t), u(t)) = f(x(t),u(t))dt + g(x(t))dw(t)
            (fx, fu, fxx, fxu, fuu, gx, gu, gxx, gxu, guu) =
                get_dynamics_jac_hess(model, X[:, t], U[:, t], t)
            fx_arr[:, :, t] = I + fx * dt
            fu_arr[:, :, t] = fu * dt
            fxx_arr[:, :, :, t] = fxx * dt
            fxu_arr[:, :, :, t] = fxu * dt
            fuu_arr[:, :, :, t] = fuu * dt
            # gx_arr[:, :, t] = I + gx * dt
            # gu_arr[:, :, t] = gu * dt
            # gxx_arr[:, :, :, t] = gxx * dt
            # gxu_arr[:, :, :, t] = gxu * dt
            # guu_arr[:, :, :, t] = guu * dt

            # compute running cost at each time step
            (l, lx, lxx, lu, luu, lxu) = instantaneous_cost(model, X[:, t], U[:, t], gradient=true)
            l_arr[t] = l * dt
            lx_arr[:, t] = lx * dt
            lxx_arr[:, :, t] = lxx * dt
            lu_arr[:, t] = lu * dt
            luu_arr[:, :, t] = luu * dt
            lxu_arr[:, :, t] = lxu * dt
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
            (fx, fu, fxx, fxu, fuu, gx, gu, gxx, gxu, guu, l, lx, lu, lxx, lxu, luu) =
                get_array_from_trajectory(
                    fx_arr, fu_arr, fxx_arr, fxu_arr, fuu_arr,
                    gx_arr, gu_arr, gxx_arr, gxu_arr, guu_arr,
                    l_arr, lx_arr, lu_arr, lxx_arr, lxu_arr, luu_arr, t
                )

            # Q = l + V
            Qx = lx + fx' * Vx
            Qu = lu + fu' * Vx
            Qxx = lxx + fx' * (Vxx + reg_param1 * I) * fx
            Qxu = lxu + fx' * (Vxx + reg_param1 * I) * fu
            Quu = luu + fu' * (Vxx + reg_param1 * I) * fu

            if !problem.islinear
                for k = 1:x_dim
                    Qxx += Vx[k] .* fxx[k, :, :]
                    Qxu += Vx[k] .* fxu[k, :, :]
                    Quu += Vx[k] .* fuu[k, :, :]
                end
            end

            # second regulation
            Quu_eig = eigen(Quu)
            if minimum(real(Quu_eig.values)) <= 0.0
                Quu += reg_param2 * I
                println("he")
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

    function _forward_pass(model::Any, problem::DDPProblem)
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
                                dp45_step(model, dynamics, t, X_new[:, t], U_new[:, t], dt) * dt

            end

            # evaluate the new trajectory
            J_new = trajectory_cost(model, X_new, U_new)

            if J_new < J_old
                problem.X, problem.U, problem.J = copy(X_new), copy(U_new), copy(J_new)
                problem.reg_param1 /= problem.reg_param1_fact / 2
                problem.reg_param2 /= problem.reg_param2_fact / 2

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

    println("> Solve DDP Problem \n")

    # X, U = copy(problem.X), copy(problem.U)
    J = copy(problem.J)

    J_old = 0
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

        _backward_pass(model, problem)
        _forward_pass(model, problem)

        J_new = problem.J
        if J_new < J_old
            success = true
            J_old = copy(J)
            J = copy(J_new)
        end
        ite += 1
    end
    println("------------------------------------------------\n\
        >>> DDP Iterations Finished <<<\n\
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
    return nothing
end