import numpy as np;
import math;

def f(x):
    x1 = x[0];
    x2 = x[1];
    return pow(x1, 4) + 4 * x1 * x2 + 2 * x2 + (pow(x2,2) / 2);

# Returns gradient at point x = [x1, x2] with function:
# f(x) = x1^4 + 4 * x1 * x2 + 2 * x2 + x2^2 / 2
# gradient of f(x) is (4 * x1 ^ 3 + 4 * x4 , 4 * x1 + 2 + x2)
def grad_f(x):
    x1 = x[0];
    x2 = x[1];
    x1_new = 4 * pow(x1, 3) + 4 * x2;
    x2_new = 4 *  x1 + 2 + x2;
    return np.array([x1_new, x2_new]);

def gradient_descent(f, grad_f, eta, x0, max_iter=100):
    xt = x0;
    print("first xt=" + str(xt));
    for i in range(1, max_iter + 1):
        eta_t = eta(i);
        grad_t = grad_f(xt);
        xt = xt - eta_t * grad_t;
        print("step=" + str(i) + " eta=" + str(eta_t) + " grad_f(xt)=" + str(grad_t) + " newxt=" + str(xt));
    return xt;

def eta_const(c = 0.1):
    def eta(t):
        return c;
    return eta;

def eta_sqrt(c = 0.1):
    def eta(t):
        return c / math.sqrt(t + 1);
    return eta;

def eta_multistep(milestones, eta_init = 0.1, c = 0.1):
    def eta(t):
        # if it passed all milestones, easy calculation
        if (t >= milestones[-1]):
            return pow(c, milestones.size) * eta_init;
        i = 0;
        p = 0;
        # if it hasn't passed all milestones, calculate the correct scalar
        while (t >= milestones[i]):
            p = p + 1;
            i = i + 1;
        return pow(c, p) * eta_init;
    return eta;
        
# Initial point (0,0)
x00 = np.array([1,1]);

x100 = gradient_descent(f, grad_f, eta_const(c = 0.01), x00, 100);
print("f(x_100) using constant step size 0.01 = " + str(f(x100)));
print("\n" * 2);

x100 = gradient_descent(f, grad_f, eta_sqrt(c = 0.1), x00, 100);
print("f(x_100) using c/sqrt(t+1) step size with c = 0.1 = " + str(f(x100)));
print("\n" * 2);

milestones = np.array([10, 60, 90]);
x100 = gradient_descent(f, grad_f, eta_multistep(milestones, eta_init = 0.1, c = 0.5), x00, 100);
print("f(x_100) using milestones="+str(milestones)+" eta policy = " + str(f(x100)));

