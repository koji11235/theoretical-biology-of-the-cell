import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import seaborn as sns





max_x = 10
x_width = 1
x = np.arange(0, max_x + 1, x_width)
max_y = 10
y_width = 1
y = np.arange(0, max_y + 1, y_width)





a = 1
b = 1
r = 1
rd = 1

dx1 = a - r * x1 * x2
dx2 = b - r * x1 * x2
dx3 = r * x1 * x2 - rd * x3


alpha = 1

max_p1 = 4
p1_width = 0.1
p1 = np.arange(0, max_p1 + p1_width, p1_width)
max_p2 = 4
p2_width = 0.1
p2 = np.arange(0, max_p2 + p2_width, p2_width)

p1, p2 = np.meshgrid(p1, p2)


dp1 = alpha / (1 + p2**2) - p1
dp2 = alpha / (1 + p1**2) - p2


plt.figure(figsize=(10, 10))
plt.title('')
plt.quiver(p1, p2, dp1, dp2, units='width')


def calculate_differential(p1, p2, alpha=1):
    dp1 = alpha / (1 + p2**2) - p1
    dp2 = alpha / (1 + p1**2) - p2
    return dp1, dp2


def calculate_trajectory(p1_0, p2_0, differential, delta=0.1, steps=1000):
    ｔrajectory = np.array([[p1_0, p2_0]])
    p1_i, p2_i = p1_0, p2_0
    for _ in range(1000):
        dp1, dp2 = differential(p1_i, p2_i)
        p1_ip1 = p1_i + delta * dp1
        p2_ip1 = p2_i + delta * dp2
        #ｔrajectory.append([p1_ip1, p2_ip1])
        ｔrajectory = np.append(
            ｔrajectory, np.array([[p1_ip1, p2_ip1]]), axis=0)
        p1_i, p2_i = p1_ip1, p2_ip1
    return ｔrajectory


p1_0, p2_0 = 0., 1.
delta = 0.1
ｔrajectory = np.empty([0, 2])
p1_i, p2_i = p1_0, p2_0
for _ in range(1000):
    dp1, dp2 = calculate_differential(p1_i, p2_i)
    p1_ip1 = p1_i + delta * dp1
    p2_ip1 = p2_i + delta * dp2
    #ｔrajectory.append([p1_ip1, p2_ip1])
    ｔrajectory = np.append(ｔrajectory, np.array([[p1_ip1, p2_ip1]]), axis=0)
    p1_i, p2_i = p1_ip1, p2_ip1


p1_0, p2_0 = 0., 1.
ｔrajectory0 = calculate_trajectory(
    p1_0, p2_0, differential=calculate_differential, delta=0.1, steps=1000)
ｔrajectory0
ｔrajectory1 = calculate_trajectory(
    4, 0.5, differential=calculate_differential, delta=0.1, steps=1000)
ｔrajectory1


plt.figure(figsize=(10, 10))
plt.title('')
plt.quiver(p1, p2, dp1, dp2, units='width')
plt.plot(ｔrajectory0[:, 0], trajectory0[:, 1])
plt.plot(ｔrajectory1[:, 0], trajectory1[:, 1])





alpha = 1

max_p1 = 4
p1_width = 0.1
p1 = np.arange(0, max_p1 + p2_width, p1_width)
max_p2 = 4
p2_width = 0.1
p2 = np.arange(0, max_p2 + p2_width, p2_width)

p1, p2 = np.meshgrid(p1, p2)


dp1, dp2 = calculate_differential(p1, p2, alpha=4)
plt.figure(figsize=(10,10))
plt.title('')
plt.streamplot(p1, p2, dp1, dp2)
#plt.plot(ｔrajectory[0], )


alpha = 1

max_p1 = 4
p1_width = 0.1
p1 = np.arange(0, max_p1 + p2_width, p1_width)
max_p2 = 4
p2_width = 0.1
p2 = np.arange(0, max_p2 + p2_width, p2_width)

p1, p2 = np.meshgrid(p1, p2)


dp1, dp2 = calculate_differential(p1, p2, alpha=alpha)
velocity = np.sqrt(dp1**2 + dp2**2)

plt.figure(figsize=(10, 10))
plt.title('')
plt.streamplot(p1, p2, dp1, dp2, color=velocity)
#plt.plot(ｔrajectory[0], )


alpha = 4

max_p1 = 4
p1_width = 0.1
p1 = np.arange(0, max_p1 + p2_width, p1_width)
max_p2 = 4
p2_width = 0.1
p2 = np.arange(0, max_p2 + p2_width, p2_width)

p1, p2 = np.meshgrid(p1, p2)


dp1, dp2 = calculate_differential(p1, p2, alpha=alpha)
velocity = np.sqrt(dp1**2 + dp2**2)

plt.figure(figsize=(10, 10))
plt.title('')
plt.streamplot(p1, p2, dp1, dp2, color=velocity)
#plt.plot(ｔrajectory[0], )





def calculate_differential(x, y, w=1):
    dx = x - w * y - (x**2 + y**2) * x
    dy = y + w * x - (x**2 + y**2) * y
    return dx, dy


max_x = 2
min_x = -2
x_width = 0.01
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 2
min_y = -2
y_width = 0.01
y = np.arange(min_y, max_y + y_width, y_width)

x, y = np.meshgrid(x, y)


dx, dy = calculate_differential(x, y, w=1)
velocity = np.sqrt(dx**2 + dy**2)

plt.figure(figsize=(10, 10))
plt.title('')
plt.streamplot(x, y, dx, dy, color=velocity)





# Parameters

A = 0.5
B = 1.5

# Range of Phase space
max_x = 2.2
min_x = -0.2
x_width = 0.001
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 3.5
min_y = -0.2
y_width = 0.001
y = np.arange(min_y, max_y + y_width, y_width)
x, y = np.meshgrid(x, y)


###########################
def calculate_trajectory(x_0, y_0, differential, delta=0.01, steps=1000):
    ｔrajectory = np.array([[x_0, y_0]])
    x_i, y_i = x_0, y_0
    for _ in range(steps):
        dx, dy = differential(x_i, y_i)
        x_ip1 = x_i + delta * dx
        y_ip1 = y_i + delta * dy
        #ｔrajectory.append([x_ip1, y_ip1])
        ｔrajectory = np.append(ｔrajectory, np.array([[x_ip1, y_ip1]]), axis=0)
        x_i, y_i = x_ip1, y_ip1
    return ｔrajectory


def calculate_differential(X, Y):
    dX = A - (B + 1) * X + X**2 * Y
    dY = B * X - X**2 * Y
    return dX, dY


x_0, y_0 = 0.8, 0.2
trajectory = calculate_trajectory(x_0, y_0,
                                  differential=calculate_differential,
                                  delta=0.1, steps=1000)
trajectory2 = calculate_trajectory(1.0, 2.0,
                                   differential=calculate_differential,
                                   delta=0.1, steps=1000)
###########################

dx, dy = calculate_differential(x, y)

velocity = np.sqrt(dx**2 + dy**2)

plt.figure(figsize=(10, 10))
plt.title('')
plt.streamplot(x, y, dx, dy, color=velocity)
#plt.scatter(x_1[np.isclose(dx_1, 0.0, atol=1e-2)],x_2[np.isclose(dx_1, 0.0, atol=1e-2)])
plt.contour(x, y, np.isclose(dx, 0.0, atol=1e-3),
            linestyles=["dashed"], colors=[sns.color_palette()[0]])
plt.contour(x, y, np.isclose(dy, 0.0, atol=1e-3),
            linestyles=["dashed"], colors=[sns.color_palette()[1]])
plt.plot(trajectory[:, 0], trajectory[:, 1], color="red")
plt.plot(trajectory2[:, 0], trajectory2[:, 1], color="blue")


t = np.arange(0, 1001)
plt.plot(t, trajectory[:, 0], color="red")
plt.plot(t, trajectory[:, 1], color="blue")


t = np.arange(0, 1001)
plt.plot(t, trajectory2[:, 0], color="red")
plt.plot(t, trajectory2[:, 1], color="blue")



def F(x, beta=8):
    return 1 / (np.exp(-beta * x) + 1)


def calculate_differential(x, gamma, theta):
    result = []
    for i, (x_i, gamma_i, theta_i) in enumerate(zip(x, gamma, theta)):
        dx_i_dt = gamma_i * \
            (F(sum([J_ij * x_j for J_ij, x_j in zip(J[i, :], x)]) -
               theta_i, beta=8) - x_i)
        result.append(dx_i_dt)
    return result


J = np.array(
    [[1, -1],
     [1, 0]])

max_x = 1
min_x = 0
x_width = 0.05
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 1
min_y = 0
y_width = 0.05
y = np.arange(min_y, max_y + y_width, y_width)

x_1, x_2 = np.meshgrid(x, y)

gamma_1 = 1
gamma_2 = 1
theta_1 = -0.1
theta_2 = 0.2

dx_1, dx_2 = calculate_differential(
    [x_1, x_2], [gamma_1, gamma_2], [theta_1, theta_2])

velocity = np.sqrt(dx_1**2 + dx_2**2)

plt.figure(figsize=(10, 10))
plt.title('')
plt.streamplot(x_1, x_2, dx_1, dx_2, color=velocity)





plt.figure(figsize=(10,10))
plt.title('')
plt.quiver(x_1, x_2, dx_1, dx_2, velocity, units='width')





# Parameters
J = np.array(
    [[1, -1],
     [1, 0]])
gamma_1 = 1
gamma_2 = 1
theta_1 = 0.1
theta_2 = 0.2
beta = 16

# Range of Phase space
max_x = 1.2
min_x = -0.2
x_width = 0.001
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 1.2
min_y = -0.2
y_width = 0.001
y = np.arange(min_y, max_y + y_width, y_width)
x_1, x_2 = np.meshgrid(x, y)


###########################
# def of sigmoid
def F(x, beta=8):
    return 1 / (np.exp(-beta * x) + 1)


def calculate_trajectory(x_0, y_0, differential, delta=0.01, steps=100000):
    ｔrajectory = np.array([[x_0, y_0]])
    x_i, y_i = x_0, y_0
    for _ in range(1000):
        dx, dy = differential(x_i, y_i)
        x_ip1 = x_i + delta * dx
        y_ip1 = y_i + delta * dy
        #ｔrajectory.append([x_ip1, y_ip1])
        ｔrajectory = np.append(ｔrajectory, np.array([[x_ip1, y_ip1]]), axis=0)
        x_i, y_i = x_ip1, y_ip1
    return ｔrajectory


def calculate_differential(x_1, x_2):
    dx_1 = gamma_1 * (F(J[0, 0] * x_1 + J[0, 1] * x_2 - theta_1, beta=8) - x_1)
    dx_2 = gamma_2 * (F(J[1, 0] * x_1 + J[1, 1] * x_2 - theta_2, beta=8) - x_2)
    return dx_1, dx_2


x_0, y_0 = 0.8, 0.2
trajectory = calculate_trajectory(x_0, y_0,
                                  differential=calculate_differential,
                                  delta=0.1, steps=1000)
trajectory
###########################


dx_1 = gamma_1 * (F(J[0, 0] * x_1 + J[0, 1] * x_2 - theta_1, beta=beta) - x_1)
dx_2 = gamma_2 * (F(J[1, 0] * x_1 + J[1, 1] * x_2 - theta_2, beta=beta) - x_2)

velocity = np.sqrt(dx_1**2 + dx_2**2)

plt.figure(figsize=(10, 10))
plt.title('')
plt.streamplot(x_1, x_2, dx_1, dx_2, color=velocity)
#plt.scatter(x_1[np.isclose(dx_1, 0.0, atol=1e-2)],x_2[np.isclose(dx_1, 0.0, atol=1e-2)])
plt.contour(x_1, x_2, np.isclose(dx_1, 0.0, atol=1e-3),
            linestyles=["dashed"], colors=[sns.color_palette()[0]])
plt.contour(x_1, x_2, np.isclose(dx_2, 0.0, atol=1e-3),
            linestyles=["dashed"], colors=[sns.color_palette()[1]])
plt.plot(trajectory[:, 0], trajectory[:, 1], color="red")


# Parameters
J = np.array(
    [[1, -1],
     [1, -0.5]])
gamma_1 = 1
gamma_2 = 1
theta_1 = -0.1
theta_2 = 0.2
beta = 8

# Range of Phase space
max_x = 1
min_x = 0
x_width = 0.001
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 1
min_y = 0
y_width = 0.001
y = np.arange(min_y, max_y + y_width, y_width)
x_1, x_2 = np.meshgrid(x, y)


###########################
def calculate_trajectory(x_0, y_0, differential, delta=0.1, steps=1000):
    ｔrajectory = np.array([[x_0, y_0]])
    x_i, y_i = x_0, y_0
    for _ in range(1000):
        dx, dy = differential(x_i, y_i)
        x_ip1 = x_i + delta * dx
        y_ip1 = y_i + delta * dy
        #ｔrajectory.append([x_ip1, y_ip1])
        ｔrajectory = np.append(ｔrajectory, np.array([[x_ip1, y_ip1]]), axis=0)
        x_i, y_i = x_ip1, y_ip1
    return ｔrajectory


def calculate_differential(x_1, x_2):
    dx_1 = gamma_1 * (F(J[0, 0] * x_1 + J[0, 1] * x_2 - theta_1, beta=8) - x_1)
    dx_2 = gamma_2 * (F(J[1, 0] * x_1 + J[1, 1] * x_2 - theta_2, beta=8) - x_2)
    return dx_1, dx_2


x_0, y_0 = 0.8, 0.2
trajectory = calculate_trajectory(x_0, y_0,
                                  differential=calculate_differential,
                                  delta=0.1, steps=1000)
trajectory
###########################

# def of sigmoid


def F(x, beta=8):
    return 1 / (np.exp(-beta * x) + 1)


dx_1 = gamma_1 * (F(J[0, 0] * x_1 + J[0, 1] * x_2 - theta_1, beta=beta) - x_1)
dx_2 = gamma_2 * (F(J[1, 0] * x_1 + J[1, 1] * x_2 - theta_2, beta=beta) - x_2)

velocity = np.sqrt(dx_1**2 + dx_2**2)

plt.figure(figsize=(10, 10))
plt.title('')
plt.streamplot(x_1, x_2, dx_1, dx_2, color=velocity)
#plt.scatter(x_1[np.isclose(dx_1, 0.0, atol=1e-2)],x_2[np.isclose(dx_1, 0.0, atol=1e-2)])
plt.contour(x_1, x_2, np.isclose(dx_1, 0.0, atol=1e-3),
            linestyles=["dashed"], colors=[sns.color_palette()[0]])
plt.contour(x_1, x_2, np.isclose(dx_2, 0.0, atol=1e-3),
            linestyles=["dashed"], colors=[sns.color_palette()[1]])
plt.plot(trajectory[:, 0], trajectory[:, 1], color="red")


def calculate_differential(x, gamma, theta):
    result = []
    for i, (x_i, gamma_i, theta_i) in enumerate(zip(x, gamma, theta)):
        dx_i_dt = gamma_i * (F(sum([J_ij * x_j for J_ij, x_j in zip(J[i,:], x)]) - theta_i, beta=8) - x_i)
        result.append(dx_i_dt)
    return result

def F(x, beta=8):
    return 1 / (np.exp(-beta * x) + 1)

J = np.array(
    [[0, -1],
     [-1, 0]])

max_x = 1
min_x = 0
x_width = 0.01
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 1
min_y = 0
y_width = 0.01
y = np.arange(min_y, max_y + y_width, y_width)

x_1, x_2 = np.meshgrid(x, y)

gamma_1 = 1
gamma_2 = 1
theta_1 = 0.5
theta_2 = 0.5

dx_1, dx_2 = calculate_differential([x_1, x_2], [gamma_1, gamma_2], [theta_1, theta_2])
                              
velocity = np.sqrt(dx_1**2 + dx_2**2)

plt.figure(figsize=(10,10))
plt.title('')
plt.streamplot(x_1, x_2, dx_1, dx_2, color=velocity)


# Parameters 
J = np.array(
    [[0, -1],
     [-1, 0]])
gamma_1 = 1
gamma_2 = 1
theta_1 = 0.5
theta_2 = 0.5

beta = 8

# Range of Phase space
max_x = 0.1
min_x = 0
x_width = 0.00005
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 0.1
min_y = 0
y_width = 0.00005
y = np.arange(min_y, max_y + y_width, y_width)
x_1, x_2 = np.meshgrid(x, y)


###########################
def calculate_trajectory(x_0, y_0, differential, delta = 0.1, steps=1000):
    ｔrajectory = np.array([[x_0, y_0]])
    x_i, y_i = x_0, y_0 
    for _ in range(1000):
        dx, dy = differential(x_i, y_i)
        x_ip1 = x_i + delta * dx
        y_ip1 = y_i + delta * dy
        #ｔrajectory.append([x_ip1, y_ip1])
        ｔrajectory = np.append(ｔrajectory, np.array([[x_ip1, y_ip1]]), axis=0)
        x_i, y_i = x_ip1, y_ip1
    return ｔrajectory

def calculate_differential(x_1, x_2):
    dx_1 = gamma_1 * (F(J[0,0] * x_1 + J[0,1] * x_2 - theta_1, beta=8) - x_1)
    dx_2 = gamma_2 * (F(J[1,0] * x_1 + J[1,1] * x_2 - theta_2, beta=8) - x_2)
    return dx_1, dx_2

x_0, y_0 = 0.8, 0.2
trajectory = calculate_trajectory(x_0, y_0, 
                                  differential=calculate_differential, 
                                  delta = 0.1, steps=1000)
trajectory
###########################

# def of sigmoid
def F(x, beta=8):
    return 1 / (np.exp(-beta * x) + 1)

dx_1 = gamma_1 * (F(J[0,0] * x_1 + J[0,1] * x_2 - theta_1, beta=beta) - x_1)
dx_2 = gamma_2 * (F(J[1,0] * x_1 + J[1,1] * x_2 - theta_2, beta=beta) - x_2)

velocity = np.sqrt(dx_1**2 + dx_2**2)

plt.figure(figsize=(10,10))
plt.title('')
plt.streamplot(x_1, x_2, dx_1, dx_2, color=velocity)
#plt.scatter(x_1[np.isclose(dx_1, 0.0, atol=1e-2)],x_2[np.isclose(dx_1, 0.0, atol=1e-2)])
plt.contour(x_1, x_2, np.isclose(dx_1, 0.0, atol=1e-4), linestyles=["dashed"], colors=[sns.color_palette()[0]])
plt.contour(x_1, x_2, np.isclose(dx_2, 0.0, atol=1e-4), linestyles=["dashed"], colors=[sns.color_palette()[1]])
plt.plot(trajectory[:,0], trajectory[:,1], color="red")
plt.xlim(0,0.1)
plt.ylim(0,0.1)


# Parameters 
J = np.array(
    [[1, 1],
     [-1, 0]])
gamma_1 = 1
gamma_2 = 1
theta_1 = 1
theta_2 = -0.5

beta = 8

# Range of Phase space
max_x = 1
min_x = 0
x_width = 0.001
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 1
min_y = 0
y_width = 0.001
y = np.arange(min_y, max_y + y_width, y_width)
x_1, x_2 = np.meshgrid(x, y)


###########################
# def of sigmoid
def F(x, beta=8):
    return 1 / (np.exp(-beta * x) + 1)

def calculate_differential(x_1, x_2):
    dx_1 = gamma_1 * (F(J[0,0] * x_1 + J[0,1] * x_2 - theta_1, beta=8) - x_1)
    dx_2 = gamma_2 * (F(J[1,0] * x_1 + J[1,1] * x_2 - theta_2, beta=8) - x_2)
    return dx_1, dx_2


def calculate_trajectory(x_0, y_0, differential, delta = 0.1, steps=1000):
    ｔrajectory = np.array([[x_0, y_0]])
    x_i, y_i = x_0, y_0 
    for _ in range(1000):
        dx, dy = differential(x_i, y_i)
        x_ip1 = x_i + delta * dx
        y_ip1 = y_i + delta * dy
        #ｔrajectory.append([x_ip1, y_ip1])
        ｔrajectory = np.append(ｔrajectory, np.array([[x_ip1, y_ip1]]), axis=0)
        x_i, y_i = x_ip1, y_ip1
    return ｔrajectory



x_0, y_0 = 0.8, 0.2
trajectory = calculate_trajectory(x_0, y_0, 
                                  differential=calculate_differential, 
                                  delta = 0.1, steps=1000)
trajectory
###########################



dx_1 = gamma_1 * (F(J[0,0] * x_1 + J[0,1] * x_2 - theta_1, beta=beta) - x_1)
dx_2 = gamma_2 * (F(J[1,0] * x_1 + J[1,1] * x_2 - theta_2, beta=beta) - x_2)

velocity = np.sqrt(dx_1**2 + dx_2**2)

plt.figure(figsize=(10,10))
plt.title('')
plt.streamplot(x_1, x_2, dx_1, dx_2, color=velocity)
plt.contour(x_1, x_2, np.isclose(dx_1, 0.0, atol=1e-3), linestyles=["dashed"], colors=[sns.color_palette()[0]])
plt.contour(x_1, x_2, np.isclose(dx_2, 0.0, atol=1e-3), linestyles=["dashed"], colors=[sns.color_palette()[1]])
plt.plot(trajectory[:,0], trajectory[:,1], color="red")


# Range of Phase space
max_x = 1
min_x = 0
x_width = 0.001
x = np.arange(min_x, max_x + x_width, x_width)
max_y = 1
min_y = 0
y_width = 0.001
y = np.arange(min_y, max_y + y_width, y_width)
x_1, x_2 = np.meshgrid(x, y)

theta_1s = [-1, -0.5, 0, 0.5, 1]
theta_2s = [-1, -0.5, 0, 0.5, 1]

fig = plt.figure(figsize=(25,25))
for i, (theta_1, theta_2) in enumerate(itertools.product(theta_1s, theta_2s)):
#     print(i, theta_1, theta_2)
#     x1 = np.random.uniform(0, 100, 20)
#     y1 = x1 * np.random.uniform(1, 2, 20)

    def calculate_differential(x_1, x_2):
        dx_1 = gamma_1 * (F(J[0,0] * x_1 + J[0,1] * x_2 - theta_1, beta=8) - x_1)
        dx_2 = gamma_2 * (F(J[1,0] * x_1 + J[1,1] * x_2 - theta_2, beta=8) - x_2)
        return dx_1, dx_2


    def calculate_trajectory(x_0, y_0, differential, delta = 0.1, steps=1000):
        ｔrajectory = np.array([[x_0, y_0]])
        x_i, y_i = x_0, y_0 
        for _ in range(steps):
            dx, dy = differential(x_i, y_i)
            x_ip1 = x_i + delta * dx
            y_ip1 = y_i + delta * dy
            ｔrajectory = np.append(ｔrajectory, np.array([[x_ip1, y_ip1]]), axis=0)
            x_i, y_i = x_ip1, y_ip1
        return ｔrajectory

    x_0, y_0 = 0.8, 0.2
    trajectory = calculate_trajectory(x_0, y_0, 
                                      differential=calculate_differential, 
                                      delta = 0.1, steps=1000)


    dx_1 = gamma_1 * (F(J[0,0] * x_1 + J[0,1] * x_2 - theta_1, beta=beta) - x_1)
    dx_2 = gamma_2 * (F(J[1,0] * x_1 + J[1,1] * x_2 - theta_2, beta=beta) - x_2)

    velocity = np.sqrt(dx_1**2 + dx_2**2)
    ax = fig.add_subplot(5, 5, i+1)
    ax.set_title(f"theta_1: {theta_1}, theta_2: {theta_2}")
    ax.streamplot(x_1, x_2, dx_1, dx_2, color=velocity)
    ax.contour(x_1, x_2, np.isclose(dx_1, 0.0, atol=1e-3), linestyles=["dashed"], colors=[sns.color_palette()[0]])
    ax.contour(x_1, x_2, np.isclose(dx_2, 0.0, atol=1e-3), linestyles=["dashed"], colors=[sns.color_palette()[1]])
    ax.plot(trajectory[:,0], trajectory[:,1], color="red")

plt.show()


theta_1s



