import numpy as np
import matplotlib.pyplot as plt
from mpi4py import  MPI

start_time = MPI.Wtime()

eps0 = 8.8541878128e-12  # 真空中的介电常数
mu0 = 1.256637062e-6  # 真空中的磁导率
c0 = 1 / np.sqrt(eps0 * mu0)  # 真空中光速
imp0 = np.sqrt(mu0 / eps0)  # 真空中的特性阻抗

jMax = 50000  # 空间离散步数
jSource = 10  # 源的位置
nMax = 2000  # 时间步数

# jMax = 500  # 空间离散步数
# jSource = 10  # 源的位置
# nMax = 2000  # 时间步数

Ex = np.zeros(jMax)  # 电场分量
Hz = np.zeros(jMax)  # 磁场分量
ExPrev = np.zeros(jMax)  # 上一时间步长的电场分量
HzPrev = np.zeros(jMax)  # 上一时间步长的磁场分量

lambdaMin = 350e-9  # meters 波长最小值
dx = lambdaMin / 20  # 空间步长
dt = dx / c0  # 时间步长

eps = np.ones(jMax) * eps0  # 电介质常数
eps[250:300] = 10 * eps0  # 材料性质

material_prof = eps > eps0  # eps0是材料区域


def sourceFunc(t):
    # return Ex[t]
    lambda_0 = 550e-9
    w0 = 2 * np.pi * c0 / lambda_0
    tau = 30
    t0 = tau * 3

    return np.exp(-(t - t0) ** 2 / tau ** 2) * np.sin(w0 * t * dt)


for n in range(nMax):
    # Update magnetic field boundaries
    Hz[jMax - 1] = HzPrev[jMax - 2]
    # Update magnetic field
    Hz[:jMax-1] = HzPrev[:jMax-1] + dt / (dx * mu0) * (Ex[1:jMax] - Ex[:jMax-1])
    HzPrev = Hz
    # Magnetic field
    Hz[jSource - 1] -= sourceFunc(n) / imp0
    HzPrev[jSource - 1] = Hz[jSource - 1]

    # Update electric field boundaries
    Ex[0] = ExPrev[1]
    # Update magnetic field
    Ex[1:] = ExPrev[1:] + dt / (dx * eps[1:]) * (Hz[1:] - Hz[:jMax-1])
    ExPrev = Ex
    # Electric field source
    Ex[jSource] += sourceFunc(n + 1)
    ExPrev[jSource] = Ex[jSource]

    # plot 连续
    if n % 10 == 0:
        plt.plot(Ex)
        plt.plot(material_prof)
        plt.ylim([-1, 1])
        # #
        # plt.ioff()
        # plt.show(block=False)
        plt.pause(0.3)
        fig = plt.gcf()
        window_id = fig.canvas.manager.num
        # window_id = fig.canvas.manager.window.winfo_id()
        # time.sleep(1)
        if plt.fignum_exists(window_id):
            plt.close()
end_time = MPI.Wtime()
print("time:", end_time - start_time)

# for debugger
# print("Ex", Ex[jMax-100:jMax])
# print("Ex length:", len(Ex))
# print("Hz", Hz)

    # # plot 不连续
    # if n % 10 == 0:
    #     plt.plot(Ex)
    #     plt.plot(material_prof)
    #     plt.ylim([-1, 1])
    #     # #
    #     # plt.ioff()
    #     plt.show()
    #     # plt.pause(0.3)
    #     # fig = plt.gcf()
    #     # window_id = fig.canvas.manager.num
    #     # window_id = fig.canvas.manager.window.winfo_id()
    #     # time.sleep(1)
    #     # if plt.fignum_exists(window_id):
    #     #     plt.close()
