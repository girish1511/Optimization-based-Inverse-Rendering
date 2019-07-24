#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.autograd
from torch.autograd import Variable
import numpy as np
from scipy.io import loadmat
from scipy.special import sph_harm
import math
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from time import time
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from torch.autograd.gradcheck import zero_gradients


# In[ ]:


# torch.device("cuda")
torch.set_default_tensor_type('torch.cuda.FloatTensor')


# In[ ]:


no_of_ver = 53215


# In[ ]:


'''Load expression principal components'''
fileName='Dataset/Coarse_Dataset/Exp_Pca.bin'
with open(fileName, mode='rb') as file: # b is important -> binary
#     fileContent = file.read()
    dim_exp = np.fromfile(file, dtype=np.int32, count=1)
    mu_exp = np.zeros(no_of_ver*3)
    base_exp = np.zeros((no_of_ver*3*dim_exp[0]), dtype=float)
    mu_exp = np.fromfile(file, dtype=np.float32, count=3*no_of_ver)
    base_exp = np.fromfile(file, dtype=np.float32, count=3*no_of_ver*dim_exp[0])

A_exp = torch.tensor(np.array(np.resize(base_exp, (no_of_ver*3, dim_exp[0]))), dtype=torch.float32, requires_grad = True)


# In[ ]:


'''Load standard deviation of expression'''
data = np.loadtxt('Dataset/Coarse_Dataset/std_exp.txt', delimiter=' ')
data = torch.tensor(data[:,np.newaxis], dtype = torch.float32,requires_grad =True )


# In[ ]:


'''Triangle mesh data and indices excluded based on 3DDFA'''
temp = loadmat('Dataset/3DDFA_Release/Matlab/ModelGeneration/model_info.mat')
temp['tri'].dtype = np.int16
trimIndex = np.array(temp['trimIndex'][:,0], dtype=np.int32)
trim_ind = np.reshape(np.array([3*trimIndex-2,3*trimIndex-1,3*trimIndex])-1,(no_of_ver*3,),'F')
tri_mesh_data = temp['tri'].T#torch.tensor(temp['tri'].T, dtype=torch.long)


# In[ ]:


def load(path):
    """takes as input the path to a .pts and returns a list of 
    tuples of floats containing the points in in the form:
    [(x_0, y_0, z_0),
     (x_1, y_1, z_1),
     ...
     (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]

    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return points


# In[ ]:


'''2D and 3D landmarks'''
lmks_2d = torch.tensor(load('Dataset/300W-Convert/300W-Original/afw/134212_1.pts'))
lmks_3d_ind = np.array(temp['keypoints'], dtype=np.int32)


# In[ ]:


print("PyTorch version: ", torch.__version__ )
print("CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)


# In[ ]:


'''Load mean shape and albedo parameters, princiapl components and standard deviations for identity and albedo'''
morph_model = loadmat('Dataset/PublicMM1/01_MorphableModel.mat')

shapePCA = morph_model['shapePC']
shapeMU = morph_model['shapeMU']
shapeSTD = morph_model['shapeEV']

texPCA = morph_model['texPC']
texMU = morph_model['texMU']
texSTD = morph_model['texEV']

p_mu = torch.tensor(shapeMU[trim_ind])
b_mu = torch.tensor(texMU[trim_ind])
A_alb = torch.tensor(texPCA[trim_ind,:100])
A_id = torch.tensor(shapePCA[trim_ind,:100])
std_id = torch.tensor(shapeSTD[:100])
std_alb = torch.tensor(texSTD[:100])
std_exp = data #Standard deviation of expression


# In[ ]:


'''To calculate Legendre Polynomial'''
def P(l, m, x):
    pmm = 1.0
    if m>0:
        somx2 = torch.sqrt((1.0-x)*(1.0+x))
        fact = 1.0
        for i in torch.arange(m):
            pmm = -fact*pmm*somx2
            fact = fact+2.0
    if l==m :
        return pmm
    pmmp1 = x*(2.0*m+1.0)*pmm
    if (l==m+1):
        return pmmp1
    pll = 0.0
    for ll in torch.arange(m+2, l+1):
        pll = ((2.0*ll-1.0)*x*pmmp1 - (ll+m-1.0)*pmm)/(ll-m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def factorial(n):
    return torch.prod(torch.arange(1,n+1), dtype=torch.float32)

def K(l, m):
    norm_const = ((2.0*l+1.0)*factorial(l-m))/((4.0*np.pi)*factorial(l+m))
    return torch.sqrt(norm_const)

'''To calculate spherical harmonics(scipy.special.sph_harm does not work with pytorch for creating computational graph)'''
def SH(m, l, phi, theta):
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
    if m==0:
        return K(l,0)*P(l,m,torch.cos(theta))
    elif m>0:
        return sqrt2*K(l,m)*torch.cos(m*phi)*P(l,m,torch.cos(theta))
    else:
        return sqrt2*K(l,-m)*torch.sin(-m*phi)*P(l,-m,torch.cos(theta))


# In[ ]:


'''To calculate rotational matrix from pitch yaw and roll'''
def rot_mat(pitch, yaw, roll):
    Ry = torch.tensor([[torch.cos(pitch),0,torch.sin(pitch)],[0,1,0],[-torch.sin(pitch),0,torch.cos(pitch)]], requires_grad = True)
    Rx = torch.tensor([[1,0,0],[0,torch.cos(roll),torch.sin(roll)],[0,-torch.sin(roll),torch.cos(roll)]], requires_grad = True)
    Rz = torch.tensor([[torch.cos(yaw),torch.sin(yaw),0],[-torch.sin(yaw),torch.cos(yaw),0],[0,0,1]], requires_grad = True)
    R = Rz@Ry@Rx
    
    return R

'''To calculate first 3 bands of spherical harmonics'''
def sh_basis(norm):
#     if torch.is_tensor(n):
#         norm = n.cpu().detach().numpy()
#     else:
#         norm = n
    theta = norm[1] #Polar angle
    phi = norm[0] #Azimuth angle
    sh = torch.zeros((9,), dtype=torch.float32, requires_grad=True)
    count = 0
    for l in torch.arange(3):
        for m in torch.arange(-l,l+1):
#             if m==0:
#                 sh[count]=np.real(sph_harm(m,l,phi,theta))
#             elif m>0:
#                 sh[count]=np.sqrt(2)*np.real(sph_harm(m,l,phi,theta))
#             else:
#                 sh[count]=np.sqrt(2)*np.imag(sph_harm(m,l,phi,theta))
#             count = count+1
                sh[count] = SH(m,l,phi,theta)
                count = count+1
            
    return sh


# In[ ]:


'''To estimate the barycentric weights of a point given that it is inside a triangle
    The barycentric weights are used later to estimate albedo by barycentric interpolation'''
def barycentric_weights(p, tri_p):
    #http://blackpawn.com/texts/pointinpoly/
    
    a = tri_p[0,:]
    b = tri_p[1,:]
    c = tri_p[2,:]
    
    v0 = c - a
    v1 = b - a
    v2 = p - a
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    denom = dot00*dot11 - dot01*dot01
    
    if denom == 0:
        u = v = 0
    else:
        u = (dot11*dot02-dot01*dot12)/denom
        v = (dot00*dot12-dot01*dot02)/denom
    
#     A = torch.stack((v0,v1)).t()
#     B = v2[:,None]
    
#     X, LU = torch.solve(B,A)
#     u = X[0,0]
#     v = X[1,0]
        
#     weights = torch.tensor([1-u-v, u, v], dtype=torch.float32, requires_grad = True)
    weights = np.array([1-u-v, u, v])
    return weights

def world_to_image(q_world, h, w):
#     temp = np.array([w/2,h/2-h+1,0])
#     q_image = (q_world + temp)*[1,-1,1]
    q_image = q_world.clone()
    
    q_image[:,0] = q_image[:,0] + w/2
    q_image[:,1] = q_image[:,1] + h/2
    q_image[:,1] = h - q_image[:,1] - 1
    
    return q_image

def rasterize(q, q_depth, tri_mesh_data, h, w):
    depth_info = -math.inf*np.ones((h,w))
#     depth_info = {}
    tri_ind_info = -np.ones((h,w))
    bary_wts_info = torch.zeros((h,w,3), requires_grad = True)
    
#     for i in range(h):
#         for j in range(w):
#             depth_info[(i,j)] = -math.inf
#             bary_wts_info[(i,j)] = 0
    q_n = q.data.cpu().numpy()
    q_depth_n = q_depth.data.cpu().numpy()
    for i in range(len(tri_mesh_data)):
        print('Rasterizing triangle: ', i+1)
        tri_ver_ind = tri_mesh_data[i,:]
        
        umin = max(int(np.ceil(np.min(q_n[tri_ver_ind, 0]))), 0) #torch.min(lmks_2d[:,0])
        umax = min(int(np.floor(np.max(q_n[tri_ver_ind, 0]))), w-1) #torch.max(lmks_2d[:,0])
        
        vmin = max(int(np.ceil(np.min(q_n[tri_ver_ind, 1]))), 0) #torch.min(lmks_2d[:,0])
        vmax = min(int(np.floor(np.max(q_n[tri_ver_ind, 1]))), h-1)
        
        if umax<umin or vmax<vmin:
            continue
        else:
            for u in range(umin, umax+1):
                for v in range(vmin, vmax+1):
                    weights = barycentric_weights([u, v], q_n[tri_ver_ind, :])
                    if (weights<0).all():
                        continue
                    else:
                        depth = np.dot(weights, q_depth_n[tri_ver_ind])
                        if depth > depth_info[v,u]:
                            depth_info[v,u] = depth
                            tri_ind_info[v,u] = i
                            bary_wts_info[v,u,:] = torch.tensor(weights, requires_grad=True)
                            
    return tri_ind_info, bary_wts_info


# In[ ]:


'''Cartesian to spherical coordinates'''
def cart2sph(n):
#     phi = np.arctan2(n[1],n[0]) #arctan(y/x)
#     theta = np.arccos(n[2]) #arccos(z)
    if torch.is_tensor(n):
        norm = n.cpu().detach().numpy()
    else:
        norm = n
    phi = np.arctan2(norm[1],norm[0])
    theta = np.arccos(norm[2])
    return torch.tensor([phi, theta], dtype=torch.float32, requires_grad = True)

'''Calculate normal of triangle for each pixel based on underlying triangle index'''
def calculate_normal(tri_ind_info, tri_mesh_data, centroid, q, h, w):
    normal_xyz = np.zeros((h, w, 3))
    normal_sph = torch.zeros((h, w, 2), requires_grad = True)
    cent = centroid.data.cpu().numpy()
    for i in range(h):
        for j in range(w):
            tri_ver = q[tri_mesh_data[tri_ind_info[i, j]-1, :],:].data.cpu().numpy()
            a = tri_ver[0,:]
            b = tri_ver[1,:]
            c = tri_ver[2,:]
            normal_xyz[i,j,:] = np.cross(a-b, b-c)/np.linalg.norm(np.cross(a-b, b-c))
            if np.dot(np.mean(tri_ver, 0)-cent, normal_xyz[i,j,:])<0:
                normal_xyz[i,j,:] *= -1
            normal_sph[i,j,:] = cart2sph(normal_xyz[i,j,:])
    return normal_sph


# In[ ]:


def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]
#     A = torch.stack((v0,v1)).t()
#     B = v2[:,None]

#     X = torch.inverse(A)@B

#     u = X[0,0]
#     v = X[1,0]
    # dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)

'''To estimate the barycentric weights of a point given that it is inside a triangle
    The barycentric weights are used later to estimate albedo by barycentric interpolation'''
def get_point_weight(point, tri_points):
    ''' Get the weights of the position
    Methods: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
     -m1.compute the area of the triangles formed by embedding the point P inside the triangle
     -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    Args:
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices(2d points) of a triangle. 
    Returns:
        w0: weight of v0
        w1: weight of v1
        w2: weight of v3
     '''
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]
    
#     A = torch.stack((v0,v1)).t()
#     B = v2[:,None]
    
#     X = torch.inverse(A)@B
#     u = X[0,0]
#     v = X[1,0]
#     dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

   
    w0 = 1 - u - v
    w1 = v
    w2 = u
    return w0,w1,w2

'''Rasterization to find underlying triangle index and barycentrics weights of each pixel'''
def rasterize_triangles(vertices, triangles, h, w):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        h: height
        w: width
    Returns:
        depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle). 
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # h, w is the size of rendering
    '''
    # initial 
    depth_buffer = torch.zeros([h, w]) - 999999. #+ torch.min(vertices[2,:]) - 999999. # set the initial z to the farest position
    triangle_buffer = torch.zeros([h, w], dtype = torch.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = torch.zeros([h, w, 3], dtype = torch.float32)  # 
    
#     for i in range(h):
#         for j in range(w):
#             depth_buffer[(i,j)] = -math.inf
#             barycentric_weight[(i,j)] = 0
    ver = vertices.data.cpu().numpy()
    for i in range(triangles.shape[0]):
        print('Rasterzing: ',i+1)
        tri = triangles[i, :] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(ver[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(ver[tri, 0]))), w-1)

        vmin = max(int(np.ceil(np.min(ver[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(ver[tri, 1]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u, v], ver[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], ver[tri, :2]) # barycentric weight
                point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                if point_depth > depth_buffer[v, u]:
                    depth_buffer[(v, u)] = point_depth
                    triangle_buffer[v, u] = i
                    barycentric_weight[v, u, :] = torch.tensor([w0, w1, w2], dtype=torch.float32, requires_grad=True)

    return triangle_buffer, barycentric_weight


# In[ ]:


'''Function to render image given 3D vertices, albedo and spherical harmonic coefficients for lighting'''
def render_color_image(q, albedo, tri_mesh_data, gamma, h, w):
    image = torch.zeros((h,w,3), dtype=torch.float32, requires_grad = True)
    alb = torch.zeros((h,w,3), dtype=torch.float32, requires_grad = True)
    centroid = torch.mean(q,0)
    
    st = time()
#     tri_ind_info, bary_wts_info = rasterize(q[:,:2], q[:,2], tri_mesh_data, h, w)
    tri_ind_info, bary_wts_info = rasterize_triangles(q, tri_mesh_data, h, w)
    print('Rasterization Done!- %f seconds' %(time()-st))
    
    n_sph = calculate_normal(tri_ind_info, tri_mesh_data, centroid, q, h, w)
    
    for i in range(h):
        for j in range(w):
            sh_func = sh_basis(n_sph[i,j,:])
            alb[i,j,:] = albedo[tri_mesh_data[tri_ind_info[i, j]-1, :],:].t()@bary_wts_info[i,j,:]
            image[i,j,:] = alb[i,j,:]*(gamma.t()@sh_func.squeeze())
            
    return image


# In[ ]:


'''Input image'''
I_in = plt.imread('Dataset/300W-Convert/300W-Original/afw/134212_1.jpg')
I_in=torch.tensor(I_in[160:384,704:928,:],dtype = torch.float32, requires_grad = True)


# In[ ]:


'''Jacobian matrix estimation'''
def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.shape[0]

    jacobian = torch.zeros((num_classes, inputs.shape[0], inputs.shape[1]))
    grad_output = torch.zeros(output.shape)
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return jacobian.squeeze()


# In[ ]:


h=w=224
chi = torch.rand(312,1,requires_grad = True, dtype=torch.float32)
# t = torch.zeros(3,1, requires_grad = True, dtype=torch.float32)
count = 1
chi_prev = chi
while True:
    print("Iteration No: ", count)
    
    al_id = chi_prev[0:100]
    al_exp = chi_prev[100:179]
    al_alb = chi_prev[179:279]
    [s, pitch, yaw, roll] = chi_prev[279:283,0]
    t = chi_prev[283:285]
    r = chi_prev[285:]
    gamma = torch.reshape(r,(3,9)).t()
    
    
    p = p_mu + torch.matmul(A_id,al_id) + torch.matmul(A_exp,al_exp)
    b = b_mu + torch.matmul(A_alb,al_alb)
    vertex = torch.reshape(p, (no_of_ver, 3))
    albedo = torch.reshape(b, (no_of_ver, 3))
    
    if count == 1:
        s = 150/(torch.max(vertex) - torch.min(vertex))
    
    R = rot_mat(pitch, yaw, roll)
    q_world = s*R@vertex.t() 
    q_world[:2,:] = q_world[:2,:] + t
    q_image = world_to_image(q_world.t(), h, w) #Convert wirld coordinates to image coordinates
    print('Rendering Image...')
    st = time()

    I_rend = render_color_image(q_image, albedo, tri_mesh_data, gamma, h, w)
    print('Rendering Done! - %f seconds' %(time()-st) )
    w_l = 100
    w_r = 5e-5
    
    E_con_r = torch.tensor([np.sqrt(1/28241)*torch.norm(I_rend - I_in)], requires_grad = True)
    E_lan_r = np.sqrt(w_l/68)*torch.norm(lmks_2d - q_image[lmks_3d_ind[0,:],:2], dim=1)
    E_reg_r = np.sqrt(w_r)*torch.cat((al_id/std_id,al_alb/std_alb,al_exp/std_exp))

    E = torch.cat((E_con_r,E_lan_r,E_reg_r[:,0]))
    J = compute_jacobian(chi_prev,E)
    chi_next = chi_prev.detach() - torch.pinverse(J.t()@J)@J.t()@E.detach().unsqueeze(1)
    increment = torch.norm(chi_next - chi_prev)
    obj_func_loss = torch.norm(E)**2
    chi_prev = chi_next.clone()
    chi_prev.requires_grad = True
    del E
    count=count+1
    print('Increment: ', increment)
    print('Loss: ', obj_func_loss)
    if obj_func_loss<10:
        break


# Gauss Newton minimizes sum of squares of residuals. E(the objective function) is considered as sum of squares of residuals. For calculating the jacobian we only need the residuals not their squares

# In[ ]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(q_world[0,:].cpu().detach(), q_world[1,:].cpu().detach(), q_world[2,:].cpu().detach())


# In[ ]:


ax.view_init(-45,45)
fig

