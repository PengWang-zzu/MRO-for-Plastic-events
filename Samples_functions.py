import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import ovito._extensions.pyscript
from   ovito.data import CutoffNeighborFinder

import ovito
from ovito.io        import import_file
from ovito.pipeline  import ReferenceConfigurationModifier
from ovito.modifiers import AtomicStrainModifier,CalculateDisplacementsModifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from   scipy.signal import find_peaks, savgol_filter
from scipy.spatial import ConvexHull
import itertools
import math
import os
import torch


import HypergraphSampling
def detectStressDrops(filename, min_drop_ratio=0.01, min_drop_magnitude=0.01):
    """
    检测应力降的位置和幅度
    
    参数:
    file(strain: 应变数据  stress: 应力数据)
    min_drop_ratio: 最小应力下降比例阈值
    min_peak_distance: 峰值之间的最小距离
    
    返回:
    stress_drops: 应力降信息列表
    """
    
    data = np.loadtxt(filename, skiprows=1)
    step=data[:, 0]
    strain = data[:, 1]
    stress = data[:, 5]

    # 寻找应力峰值（应力降开始点）
    peaks, _ = find_peaks(stress, distance=1)
    
    stress_drops = []
    
    for i, peak_idx in enumerate(peaks):
        if peak_idx >= len(stress) - 5:
            continue
            
        # 从峰值点开始向后寻找局部最小值（应力降结束点）
        search_end = peaks[i+1] if i < len(peaks)-1 else len(stress)-1
        if search_end - peak_idx < 5:
            continue
            
        # 在峰值后的区域寻找最小值
        valley_idx = peak_idx + np.argmin(stress[peak_idx:search_end])
        
        # 计算应力下降幅度
        peak_stress = stress[peak_idx]
        valley_stress = stress[valley_idx]
        drop_magnitude = peak_stress - valley_stress
        drop_ratio = drop_magnitude / peak_stress if peak_stress > 0 else 0
        
        # 只保留显著的应力降
        if drop_ratio >= min_drop_ratio or drop_magnitude > min_drop_magnitude:
            stress_drops.append({
                'step':step[valley_idx],
                'peak_index': peak_idx,
                'peak_strain': strain[peak_idx],
                'peak_stress': peak_stress,
                
                'valley_index': valley_idx,
                'valley_strain': strain[valley_idx],
                'valley_stress': valley_stress,
                
                'drop_magnitude': drop_magnitude,
                'drop_ratio': drop_ratio,
                'strain_interval': strain[valley_idx] - strain[peak_idx]
            })
    
    return stress_drops


def findAtomNeighbors(bond_array):
    """
    使用pandas快速找到每个原子的连接原子，包括原子自身
    """
    # 转换为DataFrame
    df = pd.DataFrame(bond_array, columns=['atom_i', 'atom_j'])
    
    # 创建两个方向的数据
    df_reverse = df.rename(columns={'atom_i': 'atom_j', 'atom_j': 'atom_i'})
    df_all = pd.concat([df, df_reverse], ignore_index=True)
    
    # 获取所有原子的唯一列表
    all_atoms = np.unique(bond_array.flatten())
    
    # 为每个原子添加自连接（原子与自身的连接）
    self_connections = pd.DataFrame({
        'atom_i': all_atoms,
        'atom_j': all_atoms
    })
    
    # 合并自连接和其他连接
    df_all_with_self = pd.concat([df_all, self_connections], ignore_index=True)
    # 按原子分组并获取所有连接原子（包括自身）
    result = df_all_with_self.groupby('atom_i')['atom_j'].apply(lambda x: sorted(set(x))).to_dict()
    
    return result



def getXYZ(file):
    with open(file, 'r') as f:
        data = f.readlines()
    rows=[]
    for line in data[9:]:
        parts = line.strip().split()
        rows.append(list(map(float, line.strip().split())))
    df=pd.DataFrame(rows,columns=data[8:9][0].strip().split()[2:])
    df['id'] = list(range(df.shape[0]))
    df['type'] = df['type'].astype(int)
    return df


def calculate_chi_params_from_neighbors(coordinates_df, neighbors_dict, box_size=None, angles=False):
    """
    从DataFrame坐标和邻居字典计算chi参数，支持周期性边界条件
    
    Parameters
    ----------
    coordinates_df : pandas.DataFrame
        原子坐标DataFrame，需要包含x、y、z列
    neighbors_dict : dict
        邻居字典，格式为 {原子索引: [邻居索引列表]}
    box_size : array-like, optional
        盒子尺寸 [Lx, Ly, Lz]，如果是周期性系统必须提供
    angles : bool
        是否返回角度信息
        
    Returns
    -------
    chiparams : list of arrays
        每个原子的chi参数向量
    cosines : list of lists, 可选
        每个原子的角度余弦值（仅在angles=True时返回）
    """
    
    # 定义chi参数的分箱边界
    bins = [-1.0, -0.945, -0.915, -0.755, -0.705, -0.195, 0.195, 0.245, 0.795, 1.0]
    chiparams = []
    cosines = []
    
    # 确保坐标数据是numpy数组格式
    if isinstance(coordinates_df, pd.DataFrame):
        positions = coordinates_df[['x', 'y', 'z']].values
    else:
        positions = coordinates_df
    
    # 处理每个原子
    for atom_idx in range(len(positions)):
        # 获取当前原子的位置
        pos = positions[atom_idx]
        
        # 获取邻居索引列表
        if atom_idx in neighbors_dict:
            neighbor_indices = neighbors_dict[atom_idx]
        else:
            # 如果没有找到邻居，使用空列表
            neighbor_indices = []
        
        # 计算从当前原子到每个邻居的向量（考虑周期性边界条件）
        diff_vectors = []
        valid_neighbors = []
        
        for neighbor_idx in neighbor_indices:
            # 确保邻居索引在有效范围内
            if neighbor_idx < len(positions):
                # 计算原始向量
                vec = positions[neighbor_idx] - pos
                
                # 如果提供了盒子尺寸，应用周期性边界条件
                if box_size is not None:
                    # 应用最小镜像约定
                    for dim in range(3):
                        if vec[dim] > box_size[dim] / 2:
                            vec[dim] -= box_size[dim]
                        elif vec[dim] < -box_size[dim] / 2:
                            vec[dim] += box_size[dim]
                
                diff_vectors.append(vec)
                valid_neighbors.append(neighbor_idx)
        
        # 计算所有邻居对之间的角度余弦
        costhetas = []
        combos = list(itertools.combinations(range(len(diff_vectors)), 2))
        
        for combo in combos:
            vec1 = diff_vectors[combo[0]]
            vec2 = diff_vectors[combo[1]]
            modvec1 = np.linalg.norm(vec1)
            modvec2 = np.linalg.norm(vec2)
            
            if modvec1 > 1e-10 and modvec2 > 1e-10:  # 避免除以零
                costheta = np.dot(vec1, vec2) / (modvec1 * modvec2)
                # 处理浮点误差导致略微超出[-1,1]范围的情况
                costheta = np.clip(costheta, -1.0, 1.0)
                costhetas.append(costheta)
        
        # 统计chi参数
        if costhetas:
            chivector = np.histogram(costhetas, bins=bins)[0]
        else:
            chivector = np.zeros(len(bins)-1)
            
        chiparams.append(chivector)
        
        if angles:
            cosines.append(costhetas)
    
    if angles:
        return chiparams, cosines
    else:
        return chiparams
    


def centroid_from_vertices(center_Atom, vertices, box=[74.81, 74.81, 74.81]):
    """
    
    参数:
    ----------
    center_Atom : numpy.ndarray
        中心原子坐标，用于周期边界处理
    vertices : numpy.ndarray
        形状为 (n, 3) 的数组，表示 n 个顶点的三维坐标
    box : list
        周期盒子大小 [x, y, z]
    
    返回:
    ----------  
    centroid : numpy.ndarray
        质心坐标 (3,)
    volume : float
        多面体体积
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    center_Atom = np.asarray(center_Atom, dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    
    # 1. 向量化周期边界处理
    delta = vertices - center_Atom
    half_box = box / 2.0
    
    # 使用向量化操作替代循环
    for dim in range(3):
        mask_pos = delta[:, dim] > half_box[dim]
        mask_neg = delta[:, dim] < -half_box[dim]
        vertices[mask_pos, dim] -= box[dim]
        vertices[mask_neg, dim] += box[dim]
    
    # 2. 计算凸包
    try:
        hull = ConvexHull(vertices)
    except Exception as e:
        print(f"警告: 凸包计算失败: {e}")
        return np.mean(vertices, axis=0), 0.0, None
    
    # 3. 获取所有三角形面
    simplices = hull.simplices  # 所有三角形的顶点索引
    vertices_array = vertices   # 顶点坐标
    
    # 4. 选择参考点（凸包的重心或顶点平均值）
    # 使用凸包重心的近似值，通常比顶点平均值更稳定
    if len(vertices) > 0:
        reference_point = np.mean(vertices, axis=0)
    else:
        return np.zeros(3), 0.0, hull
    
    # 5. 向量化计算四面体体积和质心
    # 获取所有三角形的顶点
    p1 = vertices_array[simplices[:, 0]]
    p2 = vertices_array[simplices[:, 1]]
    p3 = vertices_array[simplices[:, 2]]
    
    # 计算向量
    v1 = p1 - reference_point
    v2 = p2 - reference_point
    v3 = p3 - reference_point
    
    # 向量化计算有向体积: (a·(b×c))/6
    # 使用einsum加速三重积计算
    cross = np.cross(v2, v3)
    dot_products = np.einsum('ij,ij->i', v1, cross)
    tetra_volumes = np.abs(dot_products) / 6.0
    
    # 计算四面体质心: (参考点 + p1 + p2 + p3)/4
    tetra_centroids = (reference_point + p1 + p2 + p3) * 0.25
    
    # 6. 加权求和
    total_volume = np.sum(tetra_volumes)
    
    if total_volume < 1e-12:
        return np.mean(vertices, axis=0), total_volume, hull
    
    # 向量化加权质心计算
    weighted_centroid = np.einsum('i,ij->j', tetra_volumes, tetra_centroids)
    centroid = weighted_centroid / total_volume
    
    return centroid, total_volume


def compute_P(Local_xyz,Atom_Neighbors):
    Atom_P=[]
    for i in range(Local_xyz.shape[0]):
        center_Atom=Local_xyz[Local_xyz['id']==i][['x','y','z']].values[0]
        if i in Atom_Neighbors[i]:
            Atom_Neighbors[i].remove(i)
        centroid, total_volume=centroid_from_vertices(center_Atom,Local_xyz[Local_xyz['id'].isin(Atom_Neighbors)][['x','y','z']].values)
        P=round(float(np.linalg.norm(center_Atom-centroid)),4)
        Atom_P.append(P)
    df = pd.DataFrame({
    'id': list(range(len(Atom_P))),
    'P': Atom_P
})
    return df


def compute_hyperedge_index(neighbors_dict):
    num_nodes = max(neighbors_dict.keys()) + 1 if neighbors_dict else 0
    
    node_indices = []
    hyperedge_indices = []
    
    for hyperedge_idx in sorted(neighbors_dict.keys()):
        neighbors = neighbors_dict[hyperedge_idx]
        nodes_in_hyperedge = [hyperedge_idx] + neighbors
        node_indices.extend(nodes_in_hyperedge)
        hyperedge_indices.extend([hyperedge_idx] * len(nodes_in_hyperedge))
    hyperedge_index = torch.tensor([node_indices, hyperedge_indices])
    return hyperedge_index


def computeData(xyz_folder,Atom_number,Atom_volume,Atom_mass):
    Data={}
    files_names=os.listdir(xyz_folder)
    xyz_files=[xyz_folder+i for i in files_names]
    
    pipeline = ovito.io.import_file(xyz_files, multiple_frames = True)
    voronoi_modifier = ovito.modifiers.VoronoiAnalysisModifier(compute_indices = True,only_selected=False,generate_bonds=True)
    Strain_Modifier=AtomicStrainModifier(reference_frame = 0,cutoff = 3.6, output_nonaffine_squared_displacements= True,
                                         use_frame_offset=True,output_strain_tensors=True)
    Strain_Modifier.use_frame_offset = True
    Strain_Modifier.frame_offset = -1
    Strain_Modifier.affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference
    pipeline.modifiers.append(voronoi_modifier)
    pipeline.modifiers.append(Strain_Modifier)

    
    for i in range(1,len(xyz_files)):

        Local_xyz=getXYZ(xyz_files[i])
        data=pipeline.compute(i)

        #图
        bond_topology = data.particles.bonds.topology
        Atom_Neighbors=findAtomNeighbors(bond_topology[:,])
        hyperedge_index=compute_hyperedge_index(Atom_Neighbors)


        #节点特征
        voro_indices = data.particles['Voronoi Index']
        Atom_Features=Local_xyz[['id','type','x','y','z','c_PA', 'c_stress[1]', 'c_stress[2]','c_stress[3]',
                                 'c_myQ[1]', 'c_myQ[2]', 'c_myQ[3]', 'c_myQ[4]','c_myQ[5]']].reset_index(drop=True)

        #voronoi指数
        Atom_Features['Voro_indices_3']=voro_indices[:,2:3]
        Atom_Features['Voro_indices_4']=voro_indices[:,3:4]
        Atom_Features['Voro_indices_5']=voro_indices[:,4:5]
        Atom_Features['Voro_indices_6']=voro_indices[:,5:6]
        
        #局部五次对称性f5
        Atom_Features['Voro_f5']=Atom_Features['Voro_indices_5']/(Atom_Features['Voro_indices_3']+Atom_Features['Voro_indices_4']+
                                                              Atom_Features['Voro_indices_5']+Atom_Features['Voro_indices_6'])

        
        #自由体积、局部密度、配位数
        Atom_Features['voro_Volume']= data.particles['Atomic Volume']#胞腔体积
        Atom_Features['atom_volume']=Atom_Features['type'].map(Atom_volume)#原子体积
        Atom_Features['free_volume'] =1-Atom_Features['atom_volume'] / Atom_Features['voro_Volume']#自由体积
        

        Atom_Features['atom_mass']=Atom_Features['type'].map(Atom_mass)#原子密度
        Atom_Features['local density'] = Atom_Features['atom_mass'] / Atom_Features['voro_Volume']

        
        #Atom_Features = Atom_Features.drop(['voro_Volume', 'atom_volume', 'atom_mass'], axis=1)

        Atom_Features['Cavity Radius']= data.particles['Cavity Radius']
        Atom_Features['Coordination']= data.particles['Coordination']


        '''
        #多面体各项异性
        Voronoi_P=compute_P(Local_xyz,Atom_Neighbors)#dataframe 两列（id和P）
        Atom_Features = pd.merge(Atom_Features, Voronoi_P, on='id', how='inner')


        #chi参数向量
        Chi_para=calculate_chi_params_from_neighbors(Local_xyz[['x','y','z']], Atom_Neighbors, box_size=[74.82,74.82,74.82], angles=False)
        Chi_data=pd.DataFrame(Chi_para, columns=['chi_1','chi_2','chi_3','chi_4','chi_5','chi_6','chi_7','chi_8','chi_9'])
        Chi_data['id']=list(range(len(Chi_para)))
        Atom_Features = pd.merge(Atom_Features, Chi_data, on='id', how='inner')
        '''
        
        Strain_Tensor=data.particles['Strain Tensor'].array
        Strain_Tensor_pd=pd.DataFrame(Strain_Tensor, columns=['Strain_XX','Strain_YY','Strain_ZZ','Strain_XY','Strain_XZ','Strain_YZ'])
        Atom_Features=pd.concat([Atom_Features,Strain_Tensor_pd],axis=1)

        #D^2
        Atom_Features['nonaffine_squared']=data.particles['Nonaffine Squared Displacement'].array
        
        Step=int(xyz_files[i].split('_')[-1].split('.')[0])
        node_features=Atom_Features[['c_PA', 'c_stress[1]', 'c_stress[2]','c_stress[3]','Coordination',
                                     'atom_mass','atom_volume',
                                     'Strain_XX', 'Strain_YY', 'Strain_ZZ','Strain_XY', 'Strain_XZ',
                                     'Strain_YZ', 'nonaffine_squared']]
        edge_features=Atom_Features[['voro_Volume','c_myQ[1]', 'c_myQ[2]', 'c_myQ[3]', 'c_myQ[4]','c_myQ[5]',
                                     'Voro_indices_3', 'Voro_indices_4', 'Voro_indices_5','Voro_indices_6', 'Voro_f5',
                                     'free_volume', 'local density','Cavity Radius']]
        
        Data.update({Step:[hyperedge_index,node_features,edge_features]})
    return Data


def compute_Times(plasticEventTimes, windows, horizon,t_start, t_end):
    event_centered_samples ={}
    for event_time in plasticEventTimes:
        for offset in range(horizon):
            start_time = event_time - 1000*(windows) - 1000*(offset)
            end_time = start_time + 1000*(windows + horizon - 1)
            if (start_time >= t_start and end_time <= t_end):
                input_times = list(range(start_time, start_time + 1000*(windows),1000))
                target_time = list(range(start_time + 1000*(windows), start_time + 1000*(windows + horizon),1000))
                event_centered_samples[event_time]=input_times
    return event_centered_samples
'''
def extract_sub_hypergraph(hypergraph, node_list):
    """
    从大超图中提取指定节点对应的子超图
    
    参数:
    hypergraph: 2行n列的numpy数组，第一行是节点ID，第二行是超边ID
    node_list: 指定的节点ID列表
    
    返回:
    sub_hypergraph: 子超图的稠密表示
    node_mapping: 节点ID到子图节点索引的映射字典
    hyperedge_mapping: 超边ID到子图超边索引的映射字典
    """
    # 将节点列表转换为集合以便快速查找
    node_set = set(node_list)
    
    # 找到所有包含指定节点的超边
    nodes = hypergraph[0]
    hyperedges = hypergraph[1]
    
    # 找出所有包含指定节点的列
    mask = np.isin(nodes, node_list)
    
    # 提取符合条件的列
    sub_nodes = nodes[mask]
    sub_hyperedges = hyperedges[mask]
    
    sub_hypergraph=torch.stack([sub_nodes, sub_hyperedges], dim=0)
  
    return sub_hypergraph
'''



def extract_sub_hypergraph(hypergraph, node_list):
    """
    从大超图中提取指定节点对应的子超图
    
    参数:
    hypergraph: 2行n列的numpy数组，第一行是节点ID，第二行是超边ID
    node_list: 指定的节点ID列表
    
    返回:
    sub_hypergraph: 子超图的稠密表示
    node_mapping: 节点ID到子图节点索引的映射字典
    hyperedge_mapping: 超边ID到子图超边索引的映射字典
    """
    # 将节点列表转换为集合以便快速查找
    node_set = set(node_list)
    
    # 找到所有包含指定节点的超边
    nodes = hypergraph[0]
    hyperedges = hypergraph[1]
    
    # 找出所有包含指定节点的列
    mask = np.isin(nodes, node_list)
    
    # 提取符合条件的列
    sub_nodes = nodes[mask]
    sub_hyperedges = hyperedges[mask]
    
    # 创建节点映射：原始节点ID -> 新的子图索引(从0开始)
    unique_nodes = np.unique(sub_nodes)
    node_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_nodes)}
    
    # 创建超边映射：原始超边ID -> 新的子图索引(从0开始)
    unique_hyperedges = np.unique(sub_hyperedges)
    hyperedge_mapping = {int(old_id): int(new_id) for new_id, old_id in enumerate(unique_hyperedges)}
    
    # 重新映射节点ID和超边ID
    mapped_nodes = np.array([node_mapping[int(node_id)] for node_id in sub_nodes])
    mapped_hyperedges = np.array([hyperedge_mapping[int(hyperedge_id)] for hyperedge_id in sub_hyperedges])
    
    # 构建子超图
    sub_hypergraph = torch.stack([
        torch.from_numpy(mapped_nodes), 
        torch.from_numpy(mapped_hyperedges)
    ], dim=0)
    
    return sub_hypergraph, node_mapping, hyperedge_mapping




def bulit_Samples(Data,plasticEventTimes,Dmin_threshold, windows, horizon,t_start, t_end):
    Samples=[]
    event_centered_samples=compute_Times(plasticEventTimes,windows, horizon,t_start, t_end)
    for i in plasticEventTimes:
        ##子图采样,一个塑性事件采样得到四个子图
        hyperedge_index = Data[i][0]
        label=Data[i][1]['nonaffine_squared'].values
        labels=torch.tensor(np.where(label > Dmin_threshold, 1, 0))
        
        sampler = HypergraphSampling.HypergraphSampler(hyperedge_index, labels)
        samples = sampler.batch_sample(num_samples=4, target_size=2000, base_seed=42)
        
        
        for j in samples:
            step = event_centered_samples[i]#时序的时间戳
            new_hyperedge_index = j['hypergraph']['hyperedge_index']#子图的稠密矩阵
            new_node_labels= j['hypergraph']['node_labels']#节点标签
            node_map = j['hypergraph']['node_mapping']
            
            sub_hypergraph=[extract_sub_hypergraph(Data[k][0], j['node_indices'])[0] for k in step]
            node_feature=[torch.from_numpy(Data[k][1].loc[list(node_map.keys())].values) for k in step]
            edge_feature=[torch.from_numpy(Data[k][2].loc[list(node_map.keys())].values) for k in step]
            label=new_node_labels
            sub_hypergraph=[tensor.to(torch.int) for tensor in sub_hypergraph]
            Samples.append({'node_features':torch.stack(node_feature, dim=0).to(torch.float32),
                            'hyperedge_features':torch.stack(edge_feature, dim=0).to(torch.float32),
                            'hyperedge_indices':sub_hypergraph,
                            'labels':torch.tensor(label).to(torch.int)})
        
    return Samples




def bulit_Times_Samples(TotalData,plasticEventTimes,windows, horizon,t_start, t_end):
    event_centered_samples = []
    for event_time in plasticEventTimes:
        for offset in range(horizon):
            start_time = event_time - 1000*(windows) - 1000*(offset)
            end_time = start_time + 1000*(windows + horizon - 1)
            if (start_time >= t_start and end_time <= t_end):
                input_times = list(range(start_time, start_time + 1000*(windows),1000))
                target_time = list(range(start_time + 1000*(windows), start_time + 1000*(windows + horizon),1000))
                event_centered_samples.append({
                    'input_times': input_times,
                    'target_time': target_time,
                    'event_position_in_target': offset  # 塑性事件在预测窗口中的位置
                })
    Samples=[]
    for i in event_centered_samples:
        print(i)
        imput_Step=[x for x in i['input_times']]
        hyperedge_index_for_sample=[TotalData[imput_Step[i]][0] for i in range(len(imput_Step))]
        padded_tensor=pad_hyperedge_indices(hyperedge_index_for_sample)

        
        node_feature_for_sample=[TotalData[imput_Step[i]][1] for i in range(len(imput_Step))]
        node_Feature_tensor=torch.from_numpy(np.stack([df.values for df in node_feature_for_sample], axis=0))

        edge_feature_for_sample=[TotalData[imput_Step[i]][2] for i in range(len(imput_Step))]
        edge_Feature_tensor=torch.from_numpy(np.stack([df.values for df in edge_feature_for_sample], axis=0))
        
        
        target_Step=[x for x in i['target_time']]
        nodes_target=[TotalData[target_Step[i]][1]['nonaffine_squared'] for i in range(len(target_Step))]
        target_for_sample=np.stack([df.values for df in nodes_target], axis=0)
        target_tensor=torch.from_numpy(target_for_sample.squeeze())
        Samples.append({'node_features':node_Feature_tensor,
                                       'hyperedge_features':edge_Feature_tensor,
                                       'hyperedge_indices':padded_tensor,
                                       'labels':target_tensor})

    return Samples

 