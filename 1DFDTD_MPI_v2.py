import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# 基本FDTD环境变量
eps0 = 8.8541878128e-12  # 真空中的介电常数
mu0 = 1.256637062e-6  # 真空中的磁导率
c0 = 1 / np.sqrt(eps0 * mu0)  # 真空中光速
imp0 = np.sqrt(mu0 / eps0)  # 真空中的特性阻抗

jMax = 50000  # 空间离散步数
jSource = 20  # 源的位置
nMax = 2000  # 时间步数

# # for debugger
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

# 基本MPI环境
comm = MPI.COMM_WORLD
size = comm.Get_size()  # 进程数
rank = comm.Get_rank()  # 标识
start_time = MPI.Wtime()

# MPI分组
jMin_local = rank * (jMax // size)
jMax_local = (rank + 1) * (jMax // size) - 1


def sourceFunc(t):
    # return Ex[t]
    lambda_0 = 550e-9
    w0 = 2 * np.pi * c0 / lambda_0
    tau = 30
    t0 = tau * 3

    return np.exp(-(t - t0) ** 2 / tau ** 2) * np.sin(w0 * t * dt)


for n in range(nMax):
    if rank > 0:
        comm.send(Ex[jMin_local], dest=rank - 1, tag=0)

        # for debugger
        # print("rank:", rank, " sending Ex[", jMin_local, "]", Ex[jMin_local], "to previous")

    if rank < size - 1:
        Ex[jMax_local + 1] = comm.recv(source=rank + 1, tag=0)

        # for debugger
        # print("rank:", rank, " receiving Ex[", jMax_local + 1, "]", Ex[jMax_local + 1], "from next")

    # Update magnetic field boundaries
    Hz[jMax - 1] = HzPrev[jMax - 2]
    # Update magnetic field
    for j in range(jMin_local, jMax_local + 1):
        if j == jMax - 1:  # jMax-1跳过
            continue
        Hz[j] = HzPrev[j] + dt / (dx * mu0) * (Ex[j + 1] - Ex[j])
        HzPrev[j] = Hz[j]

    # for debugger
    # print("updating magnetic ", jMin_local, "to", jMax_local)
    # print("n:", n, Hz)
    # print("\n")

    # Magnetic field
    # if jMin_local <= jSource - 1 <= jMax_local:
    Hz[jSource - 1] -= sourceFunc(n) / imp0
    HzPrev[jSource - 1] = Hz[jSource - 1]

    if rank < size - 1:
        comm.send(Hz[jMax_local], dest=rank + 1)

        # for debugger
        # print("rank:", rank, " sending Hz[", jMax_local, "]", Hz[jMax_local], "to next")

    if rank > 0:
        Hz[jMin_local - 1] = comm.recv(source=rank - 1)

        # for debugger
        # print("rank:", rank, " receiving Hz[", jMin_local - 1, "]", Hz[jMin_local - 1], "from previous")

    # Update electric field boundaries
    Ex[0] = ExPrev[1]
    # Update electric field
    for j in range(jMin_local, jMax_local + 1):
        if j == 0:  # 0跳过
            continue
        Ex[j] = ExPrev[j] + dt / (dx * eps[j]) * (Hz[j] - Hz[j - 1])
        ExPrev[j] = Ex[j]

    # if jMin_local <= jSource <= jMax_local:
    Ex[jSource] += sourceFunc(n + 1)
    ExPrev[jSource] = Ex[jSource]

    # for debugger
    # print("updating electric ", jMin_local, "to", jMax_local)
    # print("n:", n, Ex)
    # print("\n")

    # # plot
    # if rank != 0:
    #     comm.send(Ex[jMin_local:jMax_local + 1], dest=0, tag=rank)
    # else:
    #     data = [[] for _ in range(5)]  # 5x?
    #     for i in range(1, size):
    #         data[i] = comm.recv(source=i, tag=i)
    #
    #     data_receive_from_other_thread = [item for sublist in data[1:] for item in sublist]
    #     data_local = Ex[jMin_local:jMax_local + 1]
    #     Ex_all = np.concatenate((data_local, data_receive_from_other_thread))
    #     # plot
    #     if n % 10 == 0:
    #         plt.plot(Ex_all)
    #         plt.plot(material_prof)
    #         plt.ylim([-1, 1])
    #         # #
    #         # plt.ioff()
    #         # plt.show(block=False)
    #         plt.pause(0.3)
    #         fig = plt.gcf()
    #         window_id = fig.canvas.manager.num
    #         # window_id = fig.canvas.manager.window.winfo_id()
    #         # time.sleep(1)
    #         if plt.fignum_exists(window_id):
    #             plt.close()
# for debugger
# if rank == 0:
#     print("rank:", rank, "Ex", Ex_all[jMax-100:jMax])
#     print("Ex_all length:", len(Ex_all))

end_time = MPI.Wtime()
if rank == 0:
    print("time:", end_time - start_time)
