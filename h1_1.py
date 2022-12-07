import numpy as np;
import math;

# the original function we are trying to minimize using gradient descent
# f(x) = x_1 ^ 4 + 4x_1x_2 + 2x_2 + (x_2 ^ 2 ) / 2
def f(x):
    x1 = x[0];
    x2 = x[1];
    return pow(x1, 4) + 4 * x1 * x2 + 2 * x2 + (pow(x2,2) / 2);

# returns gradient of f(x) at point x = [x1, x2] 
# gradient of f(x) is (4 * x1 ^ 3 + 4 * x4 , 4 * x1 + 2 + x2)
def grad_f(x):
    x1 = x[0];
    x2 = x[1];
    x1_new = 4 * pow(x1, 3) + 4 * x2;
    x2_new = 4 *  x1 + 2 + x2;
    return np.array([x1_new, x2_new]);

# gradient descent algorithm implementation
def gradient_descent(f, grad_f, eta, x0, max_iter=100, show_steps = False):
    xt = x0;
    if (show_steps):
        print("first xt=" + str(xt));
    for i in range(1, max_iter + 1):
        eta_t = eta(i);
        grad_t = grad_f(xt);
        xt = xt - eta_t * grad_t;
        if (show_steps):
            print("step=" + str(i) + " eta=" + str(eta_t) + " grad_f(xt)=" + str(grad_t) + " newxt=" + str(xt));
    xresult = f(xt);
    print("f(x_" + str(max_iter) + ")=" + str(xresult));
    return xresult;

# constant eta function constructor
def eta_const(c = 0.1):
    def eta(t):
        return c;
    return eta;

# c / sqrt(t + 1) eta function constructor
def eta_sqrt(c = 0.1):
    def eta(t):
        return c / math.sqrt(t + 1);
    return eta;

# milestone based eta function constructor
def eta_multistep(milestones, eta_init = 0.1, c = 0.1):
    def eta(t):
        p = 0;
        if (t >= milestones[-1]):
            # if it passed all milestones, easy calculation
            p = milestones.size;
        else:
            # if it hasn't passed all milestones, calculate the correct scalar
            i = 0;
            while (t >= milestones[i]):
                p = p + 1;
                i = i + 1;
        return pow(c, p) * eta_init;
    return eta;

# Initial point (1,1)
x00 = np.array([1,1]);

print("constant eta with c = 0.01");
x100 = gradient_descent(f, grad_f, eta_const(c = 0.01), x00, 100);

print("\neta calculation c / sqrt(t + 1)");
x100 = gradient_descent(f, grad_f, eta_sqrt(c = 0.1), x00, 100);

print("\nmilestone based eta");
milestones = np.array([10, 60, 90]);
x100 = gradient_descent(f, grad_f, eta_multistep(milestones, eta_init = 0.1, c = 0.5), x00, 100);

