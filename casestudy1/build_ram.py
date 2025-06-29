import numpy as np
import math
from sympy import nextprime
from scipy.sparse import bmat
from scipy.sparse import csr_matrix, block_diag
from scipy.sparse.linalg import eigsh
import numpy as np
from scipy.sparse import issparse
import numpy as np
from scipy.sparse.linalg import svds
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def is_prime(n: int) -> bool:
    """verify if the number n is prime"""
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def cyclic_shift_matrix(q: int) -> np.ndarray:
    """generate a cyclic shift matrix of size q"""
    if not is_prime(q):
        raise ValueError("q must be a prime number")
    P = np.zeros((q, q), dtype=int)
    for i in range(q):
        P[i, (i-1) % q] = 1
    return P

def validate_parameters(q: int, l: int) -> None:
    """verify the parameters q and l"""
    if not is_prime(q):
        raise ValueError("q must be a prime number")
    if l < 1 or l > q-1:
        # the critical requirement for l is still gcd(q, l) == 1
        # l < q is for simplicity, since l must be a positive integer less than q
        raise ValueError("l must be in the range [1, q-1]")

# def reshape_graph_shape(graph_shape: tuple) -> np.ndarray:
#     """reshape the input graph shape to 2 elements"""
#     """for conv layer, input channel is the product of input channel and kernel height and width"""
#     if len(graph_shape) == 2:
#         return np.array(graph_shape).reshape(2)
#     elif len(graph_shape) == 4:
#         input_size = graph_shape[1] * graph_shape[2] * graph_shape[3]
#         return np.array([graph_shape[0], input_size]).reshape(2)
#     else:
#         raise ValueError("graph_shape must be of length 2 or 4")

def find_valid_numbers(max_q: int):
    valid_numbers = []
    for q in range(2, max_q + 1):
        if is_prime(q):
            for l in range(1, q):
                try:
                    validate_parameters(q, l)
                    valid_numbers.append((q, l))
                except ValueError:validate_parameters
                    #continue
    return valid_numbers
# # Example usage
# max_q = 200
# valid_numbers = find_valid_numbers(max_q)
# print("Valid (q, l) pairs:")
# for q, l in valid_numbers:
#     print(f"q = {q}, l = {l}, left = {q**2}, right = {l*q}")

def find_ramanujan_graph_params(b: int, a: int):
    """
    Find the optimal (q, l) for constructing an undirected Ramanujan graph where:
    - One side size is q^2 >= min(a, b),
    - The other side size is q * l >= max(a, b),
    - gcd(q, l) = 1,
    and q is a prime.

    Returns:
        dict: {"q": q, "l": l, "size_1": q^2, "size_2": q*l}
    """
    def is_coprime(x, y):
        return math.gcd(x, y) == 1

    # Determine the smaller and larger target sizes
    target_small = min(a, b)
    target_large = max(a, b)

    best_q = None
    best_l = None
    best_size_1 = float('inf')
    best_size_2 = float('inf')
    best_total_size = float('inf')

    # Case 1: q^2 >= target_small, q*l >= target_large
    q_candidate = 2
    while True:
        q_squared = q_candidate ** 2
        if q_squared < target_small:
            q_candidate = nextprime(q_candidate)
            continue

        # Find minimal l such that q_candidate * l >= target_large and gcd(q_candidate, l) = 1
        min_l = math.ceil(target_large / q_candidate)
        l_candidate = min_l
        while True:
            if is_coprime(q_candidate, l_candidate):
                break
            l_candidate += 1

        size_1 = q_squared
        size_2 = q_candidate * l_candidate
        total_size = size_1 + size_2

        if total_size < best_total_size:
            best_q = q_candidate
            best_l = l_candidate
            best_size_1 = size_1
            best_size_2 = size_2
            best_total_size = total_size

        # Early termination if q^2 is significantly larger than target_small
        if q_squared > 2 * target_small and best_q is not None:
            break

        q_candidate = nextprime(q_candidate)

    # Case 2: q^2 >= target_large, q*l >= target_small (swap roles of q^2 and q*l)
    q_candidate = 2
    while True:
        q_squared = q_candidate ** 2
        if q_squared < target_large:
            q_candidate = nextprime(q_candidate)
            continue

        # Find minimal l such that q_candidate * l >= target_small and gcd(q_candidate, l) = 1
        min_l = math.ceil(target_small / q_candidate)
        l_candidate = min_l
        while True:
            if is_coprime(q_candidate, l_candidate):
                break
            l_candidate += 1

        size_1 = q_squared
        size_2 = q_candidate * l_candidate
        total_size = size_1 + size_2

        if total_size < best_total_size:
            best_q = q_candidate
            best_l = l_candidate
            best_size_1 = size_1
            best_size_2 = size_2
            best_total_size = total_size

        if q_squared > 2 * target_large and best_q is not None:
            break

        q_candidate = nextprime(q_candidate)

    # compare a,b; compare size_1, size_2
    # assign big to big, small to small

    if a > b:
        a_new = max(best_size_1, best_size_2)
        b_new = min(best_size_1, best_size_2)
    else:
        a_new = min(best_size_1, best_size_2)
        b_new = max(best_size_1, best_size_2)
    

    return {
        "q": best_q,
        "l": best_l,
        "a_new": a_new,
        "b_new": b_new,
    }

def build_biadjacency(q: int, l: int) -> np.ndarray:
    """构造 q² × lq 的双邻接矩阵 B"""
    #validate_parameters(q, l)
    P = cyclic_shift_matrix(q)
    Iq = np.eye(q, dtype=int)
    
    blocks = []
    for row_idx in range(q):            # 0 到 q-1 行块
        row_blocks = []
        for col_idx in range(l):        # 0 到 l-1 列块
            power = row_idx * col_idx   # 幂次公式：P^{row_idx * col_idx}
            if power == 0:
                block = Iq              # I_q = P^0
            else:
                block = np.linalg.matrix_power(P, power)
            row_blocks.append(block)
        # 水平堆叠当前行块的所有列块
        row_matrix = np.hstack(row_blocks)
        blocks.append(row_matrix)
    # 垂直堆叠所有行块
    B = np.vstack(blocks)
    return B

# def build_adjacency(B: np.ndarray) -> csr_matrix:
#     """构建完整的邻接矩阵 Adj (优化版)"""
#     q_sq = B.shape[0]
#     l_q = B.shape[1]
#     # 构造稀疏矩阵块
#     zero_top_left = csr_matrix((q_sq, q_sq), dtype=np.float64)
#     zero_bottom_right = csr_matrix((l_q, l_q), dtype=np.float64)
#     B_sparse = csr_matrix(B.astype(np.float64))
#     # 使用 bmat 直接构造稀疏分块矩阵
#     Adj = bmat(
#         [[zero_top_left, B_sparse],
#          [B_sparse.T, zero_bottom_right]],
#         format='csr',
#         dtype=np.float64
#     )
#     return Adj

def is_ramanujan_bipartite(B, verbose=True):
    """
    validate if the bipartite graph represented by the biadjacency matrix B is a Ramanujan graph.
    A bipartite graph is a Ramanujan graph if it is (d1, d2)-regular and the non-trivial eigenvalues
    of its adjacency matrix satisfy the condition:
    |λ| ≤ √(d1 - 1) + √(d2 - 1) for all non-trivial eigenvalues λ.
    Parameters:
        B (np.ndarray): The biadjacency matrix of the bipartite graph.
        verbose (bool): If True, prints additional information about the graph.
    Returns:
        bool: True if the graph is a Ramanujan graph, False otherwise.
    """
    B = np.asarray(B)
    m, n = B.shape
    d1 = int(np.sum(B[0]))  
    d2 = int(np.sum(B[:, 0]))  

    # verify if B is a (d1, d2)-regular bipartite graph
    if not (np.all(np.sum(B, axis=1) == d1) and np.all(np.sum(B, axis=0) == d2)):
        raise ValueError("Input matrix B is not a (d1, d2)-regular bipartite graph.")

    # Compute eigenvalues
    A = np.block([[np.zeros((m, m)), B], [B.T, np.zeros((n, n))]])
    eigenvalues = eigsh(A, k=m + n - 2, return_eigenvectors=False)  # 计算所有非平凡特征值

    # remove max and min eigen values（λ_max = √(d1*d2), λ_min = -√(d1*d2)）
    lambda_max = np.sqrt(d1 * d2)
    non_trivial_eigenvalues = [lam for lam in eigenvalues if not np.isclose(abs(lam), lambda_max)]

    # verify the condition for Ramanujan graph
    bound = np.sqrt(d1 - 1) + np.sqrt(d2 - 1)
    condition = all(abs(lam) <= bound + 1e-10 for lam in non_trivial_eigenvalues)

    if verbose:
        print(f"- left d1: {d1}, right d2: {d2}")
        print(f"- Theo: √(d1-1) + √(d2-1) = {bound:.4f}")
        # print max eigenvalue after removing trivial eigenvalues in non_trivial_eigenvalues
        #print(f"- 非平凡特征值数量: {len(non_trivial_eigenvalues)}")
        if len(non_trivial_eigenvalues) > 0:
            max_non_trivial_eigenvalue = max(abs(lam) for lam in non_trivial_eigenvalues)
            print(f"- max_non_trivial_eigenvalue: {max_non_trivial_eigenvalue:.4f}")
        else:
            max_non_trivial_eigenvalue = 0
            print("- no non-trivial eigenvalues found")
        #print(f"- 非平凡特征值 (去除了 ±{lambda_max:.4f}): {non_trivial_eigenvalues}")
        #print(f"- Ram: {condition}")

    return condition

def center_crop(matrix, target_shape):
    """
    Crop the matrix from the center to the target size (b, a), discarding the edges.

    Parameters:
    matrix: Input matrix, shape (b_new, a_new).
    target_shape: Target size (b, a).

    Returns:
    The cropped matrix, shape (b, a).
    """
    b_newl, a_newl = matrix.shape
    b, a = target_shape
    
    # 检查目标尺寸是否合法
    if b > b_newl or a > a_newl:
        raise ValueError(f"Target shape {target_shape} must smaller than ori {(b_newl, a_newl)}")
    
    # 计算起始和结束索引
    start_row = (b_newl - b) // 2
    start_col = (a_newl - a) // 2
    end_row = start_row + b
    end_col = start_col + a
    
    # 裁剪矩阵
    cropped_matrix = matrix[start_row:end_row, start_col:end_col]
    
    return cropped_matrix

def restore_conv_weight(flattened_weight, original_shape, sparse_format=False):
    """
    将展开的二维稀疏矩阵恢复为四维卷积权重。
    
    参数:
        flattened_weight: 二维矩阵 [cout, cin*kn*kw]，可以是 NumPy 数组、PyTorch 张量或稀疏张量。
        original_shape: 原始四维形状 [cout, cin, kn, kw]。
        sparse_format: 是否返回 PyTorch 稀疏张量（仅当输入为 PyTorch 张量时有效）。
        
    返回:
        恢复后的四维张量 [cout, cin, kn, kw]（密集或稀疏格式）。
    """
    cout_flatten, cin_kn_kw = flattened_weight.shape
    cout, cin, kn, kw = original_shape
    
    # 检查形状是否兼容
    assert cout_flatten == cout, "输出通道数 cout 不匹配！"
    assert cin_kn_kw == cin * kn * kw, "展开维度 cin*kn*kw 不匹配！"
    
    # 如果是 PyTorch 稀疏张量，先转换为密集格式以简化操作
    if isinstance(flattened_weight, torch.Tensor) and flattened_weight.is_sparse:
        flattened_weight = flattened_weight.to_dense()
    
    # 恢复为四维张量
    if isinstance(flattened_weight, np.ndarray):
        restored_weight = flattened_weight.reshape(cout, cin, kn, kw)
    elif isinstance(flattened_weight, torch.Tensor):
        restored_weight = flattened_weight.view(cout, cin, kn, kw)
    else:
        raise TypeError("输入必须是 NumPy 数组或 PyTorch 张量！")
    
    # 转换为稀疏格式（如果需要）
    # if sparse_format and isinstance(flattened_weight, torch.Tensor):
    #     restored_weight = restored_weight.to_sparse()
    
    return restored_weight




def build_ram(graph_shape = (64, 64, 2, 2)):
    # input required shapes
    # graph_shape_fc = (5, 5) # output channel, input channel
    # graph_shape_conv = (5, 5, 3, 3) # output channel, input channel, kernel height, kernel width

    #graph_shape = (64, 64, 2, 2) # output channel b, input channel a


    if len(graph_shape) == 2:
        b = graph_shape[0] # output channel
        a = graph_shape[1] # input channel
        is_conv = False
    elif len(graph_shape) == 4:
        b = graph_shape[0] 
        a = graph_shape[1] * graph_shape[2] * graph_shape[3]
        output_channel = graph_shape[0] # output channel
        input_channel = graph_shape[1] # input channel
        kernel_height = graph_shape[2]
        kernel_width = graph_shape[3]
        is_conv = True
    else:
        raise ValueError("graph_shape must be of length 2 or 4")
    
    converted_shape = (b, a) 
    print(f"Input graph shape: {graph_shape}, b_output: {b}, a_input: {a}")
    result = find_ramanujan_graph_params(b, a)
    # print(result)  

    # construct ramanujan graph
    q = result["q"]
    l = result["l"]
    a_new = result["a_new"]
    b_new = result["b_new"]
    print(f"q: {q}, l: {l}, a_new: {a_new}, b_new: {b_new}")
    B = build_biadjacency(q, l)
    print(f"Biadjacency matrix shape: {B.shape}")
    # Tranpose B to get the biadjacency matrix
    #B = B.T  # Transpose to get the biadjacency matrix
    if(not q*q == b_new):
        B = B.T

    is_ramanujan = is_ramanujan_bipartite(B, True)
    print(f"Ram: {is_ramanujan}")
    print(f"Adjacency matrix shape: {B.shape}")
    #print("full ramanujan biadjacency matrix B:")
    #print(B)
    # cut the matrix to the required shape
    B_exact = center_crop(B, converted_shape)
    print(f"Cropped Ram")
    print(B_exact.shape)
    if is_conv:
        # reshape to (b, a, kernel_height, kernel_width)
        B_exact = restored_np = restore_conv_weight(B_exact, graph_shape)
        print(f"Reshaped Ram to conv shape: {B_exact.shape}")
    else:
        # reshape to (b, a)
        B_exact = B_exact.reshape((b, a))
        print(f"Reshaped Ram to fc shape: {B_exact.shape}")
    #print(B_exact)
    mask = B_exact
    return mask

def draw_bipartite_adjacency_graph(adj_matrix, ramanujan_status="N/A"):
    """
    根据给定的二维邻接矩阵绘制左右连接图（bipartite graph）。

    参数：
    - adj_matrix: 2D numpy array，邻接矩阵，大小为 [l, q]
    - ramanujan_status: str，用于标注是否为Ramanujan图
    """
    l, q = adj_matrix.shape

    G = nx.DiGraph()
    
    # 创建左右节点
    left_nodes = [f"L{i}" for i in range(l)]
    right_nodes = [f"R{j}" for j in range(q)]
    G.add_nodes_from(left_nodes)
    G.add_nodes_from(right_nodes)

    # 添加边并记录边列表
    edges = []
    for i in range(l):
        for j in range(q):
            if adj_matrix[i][j] == 1:
                G.add_edge(left_nodes[i], right_nodes[j])
                edges.append((left_nodes[i], right_nodes[j]))

    # 布局位置
    pos = {}
    for i, node in enumerate(left_nodes):
        pos[node] = (-1, -i)
    for j, node in enumerate(right_nodes):
        pos[node] = (1, -j)

    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 左侧节点
    nx.draw_networkx_nodes(
        G, pos, nodelist=left_nodes, node_color='tomato', node_size=300,
        edgecolors='black', linewidths=1, label=f'Left Nodes ({l})'
    )

    # 右侧节点
    nx.draw_networkx_nodes(
        G, pos, nodelist=right_nodes, node_color='dodgerblue', node_size=300,
        edgecolors='black', linewidths=1, label=f'Right Nodes ({q})'
    )

    # 边
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=1.5, edge_color='gray', alpha=0.7
    )

    # 文字标注
    plt.text(0.5, -0.2,
             f"• q = {q}, l = {l}\n"
             f"• Ramanujan = {ramanujan_status}\n"
             f"• Left Degree ≈ {np.round(np.sum(adj_matrix, axis=1).mean(), 2)}, "
             f"Right Degree ≈ {np.round(np.sum(adj_matrix, axis=0).mean(), 2)}\n"
             f"• Edges = {len(edges)}",
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.axis('off')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.title("Biregular Graph Visualization", fontsize=12, pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #mask = build_ram(graph_shape = (64, 64, 2, 2))
    mask = build_ram(graph_shape = (20,20))
    print("Final mask shape:", mask.shape)
    print(mask)
    # draw_bipartite_adjacency_graph(mask, ramanujan_status="N/A")