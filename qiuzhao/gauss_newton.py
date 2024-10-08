import numpy as np

def gauss_newton(x, y, theta_init, max_iters=100, tol=1e-6):
    """
    使用高斯-牛顿法拟合参数 theta.
    
    参数:
    x : 自变量数组
    y : 因变量数组 (观测值)
    theta_init : 初始参数猜测值
    max_iters : 最大迭代次数
    tol : 收敛阈值
    
    返回:
    theta : 拟合的参数
    """
    theta = theta_init
    m = len(x)  # 样本点数量

    for _ in range(max_iters):
        # 计算误差 e = f(x, theta) - y
        residuals = (theta[0] + theta[1] * x) - y
        
        # 雅可比矩阵 J (对于线性模型 f(x, theta) = theta0 + theta1 * x )
        J = np.zeros((m, 2))
        J[:, 0] = 1  # 对 theta0 的偏导
        J[:, 1] = x  # 对 theta1 的偏导
        
        # 更新步长: Δtheta = (J^T * J)^(-1) * J^T * residuals
        J_T_J = J.T @ J
        if np.linalg.det(J_T_J) == 0:
            print("雅可比矩阵不可逆，无法继续更新")
            break

        delta_theta = np.linalg.inv(J_T_J) @ J.T @ residuals
        
        # 更新参数 theta
        theta = theta - delta_theta

        # 检查是否收敛
        if np.linalg.norm(delta_theta) < tol:
            print(f"收敛于第 {_+1} 次迭代")
            break
            
    return theta

# 示例数据
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 7, 13, 21])  # 假设的非线性数据

# 初始猜测值
theta_init = np.array([0.0, 1.0])

# 运行高斯-牛顿法
theta_final = gauss_newton(x, y, theta_init)

print(f"拟合后的参数: theta0 = {theta_final[0]:.4f}, theta1 = {theta_final[1]:.4f}")
