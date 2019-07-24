#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autograd.numpy as np
import sys
import math
from scipy.io import loadmat
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from autograd import jacobian, grad
import autograd
from joblib import Parallel, delayed
import multiprocessing


# In[2]:


class model_fitting(object):
    def __init__(self, h, w, use_gpu=True):
        self.no_of_ver = 53215
        self.h = h
        self.w = w
        self.use_gpu = use_gpu
        self.p_mu = None
        self.b_mu = None
        self.A_alb = None
        self.A_id = None
        self.A_exp = None
        self.std_id = None
        self.std_alb = None
        self.std_exp = None
        self.chi = np.random.randn(312,1)
        self.chi_final = None
        self.I_in = None
        self.I_rend = None
        self.tri_mesh_data = None
        self.lmks={}
        self.no_of_lmks = None
        self.no_of_face_pixels = None
        self.verteex = None
        self.albedo = None
        self.J = None
        self.load_data()
        
        
    '''Function to load .pts file. Landmarks for 300W dataset are stored in .pts format'''
    def load(self,path):
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
    
    
    def load_data(self):
        '''Function to load all the necessary data
        Prinicpal components, standard deviations, input image, triangle mesh data, landmarks'''
        
        #Expression parameters
        fileName='Dataset/Coarse_Dataset/Exp_Pca.bin'
        with open(fileName, mode='rb') as file: # b is important -> binary
        #     fileContent = file.read()
            dim_exp = np.fromfile(file, dtype=np.int32, count=1)
            mu_exp = np.zeros(self.no_of_ver*3)
            base_exp = np.zeros((self.no_of_ver*3,dim_exp[0]), dtype=float)
            mu_exp = np.fromfile(file, dtype=float, count=3*self.no_of_ver)
            base_exp = np.fromfile(file, dtype=float, count=3*self.no_of_ver*dim_exp[0])
        self.A_exp = np.array(np.resize(base_exp, (self.no_of_ver*3, dim_exp[0])))
        
        data = np.loadtxt('Dataset/Coarse_Dataset/std_exp.txt', delimiter=' ')
        data=data[:,np.newaxis]
        self.std_exp = np.array(data)
        
        #Triangle mesh data
        temp = loadmat('Dataset/3DDFA_Release/Matlab/ModelGeneration/model_info.mat')
        trimIndex = np.array(temp['trimIndex'][:,0], dtype=np.int32)
        trim_ind = np.reshape(np.array([3*trimIndex-2,3*trimIndex-1,3*trimIndex])-1,(self.no_of_ver*3,),'F')#np.append(3*trimIndex-2,np.append( 3*trimIndex-1, 3*trimIndex))
        self.tri_mesh_data = temp['tri'].T - 1
        
        #3D and 2D landmarks data
        lmks_3d_ind = temp['keypoints']
        lmks_2d = np.array(self.load('Dataset/300W-Convert/300W-Original/afw/134212_1.pts'))
        self.no_of_lmks = len(lmks_2d)
        self.lmks['2d'] = lmks_2d-[700,144]
        self.lmks['3d'] = lmks_3d_ind
        self.no_of_face_pixels = len(lmks_3d_ind)
        
        #Identity and Albedo parameters
        morph_model = loadmat('Dataset/PublicMM1/01_MorphableModel.mat')
        shapePCA = morph_model['shapePC']
        shapeMU = morph_model['shapeMU']
        shapeSTD = morph_model['shapeEV']

        texPCA = morph_model['texPC']
        texMU = morph_model['texMU']
        texSTD = morph_model['texEV']
        
        self.p_mu = shapeMU[trim_ind]
        self.b_mu = texMU[trim_ind]
        self.A_alb = np.array(texPCA[trim_ind,:100])
        self.A_id = np.array(shapePCA[trim_ind,:100])
        self.std_id = np.array(shapeSTD[:100])
        self.std_alb = np.array(texSTD[:100])
        
        #Input image
        I_in = plt.imread('Dataset/300W-Convert/300W-Original/afw/134212_1.jpg')
        self.I_in=I_in[144:400,700:956,:]
        
        #Approximate estimation of number of face pixels using first 17 landmarks
        polygon = Polygon(self.lmks['2d'][:17,:])
        temp2 = np.empty((self.h,self.w))
        for i in range(self.h):
            for j in range(self.w):
                point = Point(i,j)
                temp2[i,j] = polygon.contains(point)
        self.no_of_face_pxls = np.sum(temp2==1)
    
    '''To calculate Associated Legendre Polynomial'''
    def P(self, l, m, x):
        pmm = 1.0
        if m>0:
            somx2 = np.sqrt((1.0-x)*(1.0+x))
            fact = 1.0
            for i in range(m):
                pmm = -fact*pmm*somx2
                fact = fact+2.0
        if l==m :
            return pmm
        pmmp1 = x*(2.0*m+1.0)*pmm
        if (l==m+1):
            return pmmp1
        pll = 0.0
        for ll in range(m+2, l+1):
            pll = ((2.0*ll-1.0)*x*pmmp1 - (ll+m-1.0)*pmm)/(ll-m)
            pmm = pmmp1
            pmmp1 = pll
        return pll

    def factorial(self,n):
        return np.prod(range(1,n+1))

    def K(self, l, m):
        norm_const = ((2.0*l+1.0)*self.factorial(l-m))/((4.0*np.pi)*self.factorial(l+m))
        return np.sqrt(norm_const)
    
    '''To calculate spherical harmonics(since scipy.special.sph_harm does not work with autograd.numpy)'''
    def SH(self, m, l, phi, theta):
        '''http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf'''
        sqrt2 = np.sqrt(2.0)
        if m==0:
            return self.K(l,0)*self.P(l,m,np.cos(theta))
        elif m>0:
            return sqrt2*self.K(l,m)*np.cos(m*phi)*self.P(l,m,np.cos(theta))
        else:
            return sqrt2*self.K(l,-m)*np.sin(-m*phi)*self.P(l,-m,np.cos(theta))
    
    '''To calculate first 3 bands of spherical harmonics'''
#     def sh_basis_scipy(self, n):
#         theta = n[1] #Polar angle
#         phi = n[0] #Azimuth angle
#         sh = np.zeros((9,), dtype=np.float32)
#         count = 0
#         for l in range(3):
#             for m in np.arange(-l,l+1):
#                 if m==0:
#                     sh[count]=np.real(sph_harm(m,l,phi,theta))
#                 elif m>0:
#                     sh[count]=np.sqrt(2)*np.real(sph_harm(m,l,phi,theta))
#                 else:
#                     sh[count]=np.sqrt(2)*np.imag(sph_harm(m,l,phi,theta))
#                 count = count+1
            
#         return sh
    
    def sh_basis(self,n):
        theta = n[1] #Polar angle
        phi = n[0] #Azimuth angle
        sh = np.zeros((9,), dtype=np.float32)
        count = 0
        return np.array([self.SH(m,l,phi,theta) for l in range(3) for m in range(-l,l+1) ])
    
    '''To calculate rotational matrix from pitch yaw and roll'''
    def rot_mat(self, pitch, yaw, roll):
        Rx = np.array([[1,0,0],
                       [0,np.cos(roll),-np.sin(roll)],
                       [0,np.sin(roll),np.cos(roll)]])
        Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                       [0,1,0],
                       [-np.sin(pitch),0,np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                       [np.sin(yaw),np.cos(yaw),0],
                       [0,0,1]])
        R = Rz@Ry@Rx

        return R
    
    '''To convert world coordinates to image coordinates'''
    def world_to_image(self, q_world):
        temp_q = np.array(q_world, copy=True)
        temp = np.array([self.w/2,self.h/2-self.h+1,0])
        q_image = (q_world + temp)*[1,-1,1]

    #     q_image[:,0] = q_image[:,0] + w/2
    #     q_image[:,1] = q_image[:,1] + h/2
    #     q_image[:,1] = h - q_image[:,1] - 1

        return q_image
    
    '''Cartesian coordinates to spherical coordinates'''
    def cart2sph(self, n):
        temp = n[1]/n[0]

        if n[0]==0:
            if n[1]<0:
                phi = -np.pi/2
            else:
                phi = np.pi/2
        else:
            if n[0]>0:
                phi = np.arctan(temp)
            elif n[1]<0:
                phi = np.arctan(temp) - np.pi
            else:
                phi = np.arctan(temp) + np.pi
    #     phi = np.arctan() #arctan(y/x)
        theta = np.arccos(n[2]) #arccos(z)

        return [phi, theta]
    
    '''Calculate normal of triangle for each pixel based on underlying triangle index'''
    def calculate_normal(self, tri_ind_info, centroid, q):
    #     normal_xyz = np.zeros((h, w, 3))
    #     normal_sph = np.zeros((h, w, 2))
        normal_xyz = {}
        normal_sph = {}

        for i in range(self.h):
            for j in range(self.w):
                normal_xyz[(i,j)] = 0
                normal_sph[(i,j)] = 0
                tri_ver = q[self.tri_mesh_data[tri_ind_info[i, j]-1, :],:]
                a = tri_ver[0,:]
                b = tri_ver[1,:]
                c = tri_ver[2,:]
                normal_xyz[(i,j)] = np.cross(a-b, b-c)/np.linalg.norm(np.cross(a-b, b-c))
                if np.dot(np.mean(tri_ver, 0)-centroid, normal_xyz[(i,j)])<0:
                    normal_xyz[(i,j)] *= -1
                normal_sph[(i,j)] = self.cart2sph(normal_xyz[(i,j)])
        return normal_sph
    
    def isPointInTri(self, point, tri_points):
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

        # dot products
        dot00 = np.dot(v0.T, v0)
        dot01 = np.dot(v0.T, v1)
        dot02 = np.dot(v0.T, v2)
        dot11 = np.dot(v1.T, v1)
        dot12 = np.dot(v1.T, v2)

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
    def get_point_weight(self, point, tri_points):
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

        # dot products
        dot00 = np.dot(v0.T, v0)
        dot01 = np.dot(v0.T, v1)
        dot02 = np.dot(v0.T, v2)
        dot11 = np.dot(v1.T, v1)
        dot12 = np.dot(v1.T, v2)

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

        return w0, w1, w2

    '''Rasterization to find underlying triangle index and barycentrics weights of each pixel'''
    def rasterize_triangles(self, vertices):
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
        depth_buffer = {}#np.zeros([h, w]) - 999999. #+ np.min(vertices[2,:]) - 999999. # set the initial z to the farest position
        triangle_buffer = np.zeros([self.h, self.w], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
        barycentric_weight = {}#np.zeros([h, w, 3], dtype = np.float32)  # 

        for i in range(self.h):
            for j in range(self.w):
                depth_buffer[(i,j)] = -math.inf
                barycentric_weight[(i,j)] = np.array([0, 0, 0])

        for i in range(self.tri_mesh_data.shape[0]):
    #         print('Rasterzing: ',i+1)
            tri = self.tri_mesh_data[i, :] # 3 vertex indices

            # the inner bounding box
            umin = max(int(np.ceil(np.min(vertices[tri, 0]))), 0)
            umax = min(int(np.floor(np.max(vertices[tri, 0]))), self.w-1)

            vmin = max(int(np.ceil(np.min(vertices[tri, 1]))), 0)
            vmax = min(int(np.floor(np.max(vertices[tri, 1]))), self.h-1)

            if umax<umin or vmax<vmin:
                continue

            for u in range(umin, umax+1):
                for v in range(vmin, vmax+1):
                    if not self.isPointInTri([u,v], vertices[tri, :2]): 
                        continue
                    w0, w1, w2 = self.get_point_weight([u, v], vertices[tri, :2]) # barycentric weight
                    point_depth = w0*vertices[tri[0], 2] + w1*vertices[tri[1], 2] + w2*vertices[tri[2], 2]
                    if point_depth > depth_buffer[v, u]:
                        depth_buffer[(v, u)] = point_depth
                        triangle_buffer[v, u] = i
                        barycentric_weight[(v, u)] = np.array([w0, w1, w2])

        return depth_buffer, triangle_buffer, barycentric_weight
    
    '''Function to render image given 3D vertices, albedo and spherical harmonic coefficients for lighting'''
    def render_color_image(self, q, albedo, gamma):
    #     image = np.zeros((h,w,3), dtype=np.float32)
        centroid = np.mean(q,0)

    #     tri_ind_info, bary_wts_info = rasterize(q[:,:2], q[:,2], tri_mesh_data, h, w)
        st = time()
        depth_info, tri_ind_info, bary_wts_info = self.rasterize_triangles(q)
        print(time()-st)

        n_sph = self.calculate_normal(tri_ind_info, centroid, q)

    #     for i in range(h):
    #         for j in range(w):
    #             sh_func = sh_basis(n_sph[(i,j)])
    #             alb[i,j,:] = (albedo[tri_mesh_data[tri_ind_info[i, j], :],:].T@bary_wts_info[(i,j)])*(gamma.T@sh_func.squeeze())
    #             image[i,j,:] = alb[i,j,:]*(gamma.T@sh_func.squeeze())
    
        image = np.array([[(albedo[self.tri_mesh_data[tri_ind_info[i, j], :],:].T@bary_wts_info[(i,j)])*(gamma.T@self.sh_basis((n_sph[(i,j)])).squeeze()) for j in range(self.w)] for i in range(self.h)])
        return image
    
    '''To calculate vertex and albedo from prinipal components and mean shape and albedo parameters'''
    def cal_ver_alb(self, al_id, al_exp, al_alb):
        p = self.p_mu + self.A_id@al_id + self.A_exp@al_exp
        b = self.b_mu + self.A_alb@al_alb
        self.vertex = np.reshape(p, (self.no_of_ver, 3))
        self.albedo = np.reshape(b, (self.no_of_ver, 3))
    
    '''To calculate the objective function as mentioned in the paper'''
    def E(self, chi):
        al_id = chi[0:100]
        al_exp = chi[100:179]
        al_alb = chi[179:279]
        [s, pitch, yaw, roll] = chi[279:283,0]
        t = chi[283:285]
        r = chi[285:]
        gamma = np.reshape(r,(9,3),'F')
        lmks_2d = self.lmks['2d']
        lmks_3d_ind = self.lmks['3d']

        R = self.rot_mat(pitch, yaw, roll)

        self.cal_ver_alb(al_id, al_exp, al_alb)
        q_world = s*R@self.vertex.T
        q_world[:2,:] = q_world[:2,:] + t
        q_image = self.world_to_image(q_world.T)

        I_rend = self.render_color_image(q_image, self.albedo, gamma)
        self.I_rend = I_rend

        w_l = 10
        w_r = 5e-5
        E_con = (1/self.no_of_face_pxls)*np.linalg.norm(I_rend - self.I_in)**2 #No of face pixels is apporximately 28241
        E_lan = (1/self.no_of_lmks)*np.linalg.norm(lmks_2d - q_image[lmks_3d_ind[0,:],:2])**2 #68 landmarks
        E_reg = np.linalg.norm(al_id/self.std_id)**2 + np.linalg.norm(al_alb/self.std_alb)**2 + np.linalg.norm(al_exp/self.std_exp)**2
        #Gauss Newton minimizes sum of squares of residuals. E(the objective function) is considered as sum of squares of residuals. For calculating the jacobian we only need the residuals not their squares
        E_con_r = np.sqrt(1/self.no_of_face_pxls)*np.linalg.norm(I_rend-self.I_in)#np.reshape((I_rend-self.I_in),-1)#E_con + w_l*E_lan + w_r*E_reg
        E_lan_r = np.sqrt(w_l/self.no_of_lmks)*np.linalg.norm(lmks_2d - q_image[lmks_3d_ind[0,:],:2], axis=1)#np.reshape((lmks_2d - q_image[lmks_3d_ind[0,:],:2]),-1)
        E_reg_r = np.sqrt(w_r)*np.concatenate(((al_id/self.std_id),(al_alb/self.std_alb),(al_exp/self.std_exp)), axis = 0)
        
        return np.concatenate((np.array([E_con_r]),E_lan_r,E_reg_r[slice(None),0]), axis=0)
    
    '''Jacobian matrix estimation'''
#     def Jaco(self, func, x, dx=10^-8):
#         '''https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python'''
        
#         n = len(x)
#         m = len(E_val)
#         jac = np.zeros((m, n))
#         for j in range(n): #through columns to allow for vector addition
#             Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
#             x_plus = np.array([(xi if k != j else xi+Dxj) for k, xi in enumerate(x)])
#             jac[:, j] = (self.E(x_plus)-func)/Dxj
#         return jac
    
    def jacob(self,j,f,x,func,dx):
#         print('x:',j)
        Dxj = (abs(x[j])*dx if x[j] != 0 else dx)
    #     x_plus = np.array([(xi if k != j else xi+Dxj) for k, xi in enumerate(x)]) 
        x_plus = x.copy()
        x_plus[j] = x_plus[j] + Dxj
        return (f(x_plus)-func)/Dxj
    
    def Gauss_Newton_optim(self):
        chi_prev = self.chi.copy()
#         jacobian_E = jacobian(self.E)
        count = 1
        num_cores = multiprocessing.cpu_count()
        n = len(chi_prev)
        dx = 1e-8
        while True:
    #     chi_prev[279] = 100/(np.max(vertex) - np.min(vertex))
            
            print("Iteration No: ", count)
            if count==1:
                self.cal_ver_alb(self.chi[0:100],self.chi[100:179],self.chi[179:279])
                chi_prev[279,0] = 150/(np.max(self.vertex)-np.min(self.vertex))
                E_val = self.E(chi_prev)
            print('s:',chi_prev[279,0])
#             self.J = jacobian_E(chi_prev)
            st=time()
            results = np.array([Parallel(n_jobs=num_cores)(delayed(self.jacob)(j,obj.E,chi_prev,E_val,dx) for j in range(n))])
            print('Parallelization time:', time()-st)
            self.J = results.squeeze().T
            chi_next = chi_prev - np.linalg.pinv(self.J.T@self.J)@self.J.T@E_val[:,np.newaxis]
            E_val = self.E(chi_next)
            print('Increment: ', np.linalg.norm(chi_next-chi_prev))
            print('Loss: ',np.linalg.norm(E_val)**2)
            chi_prev = chi_next
            count=count+1
            if np.linalg.norm(E_val)**2<100:
                self.chi_final = chi_prev
                break

    def plot_rendered_image(self):
        im = self.I_rend+np.abs(np.min(self.I_rend))
        im = im/np.max(im)
        plt.imshow(im)


# In[3]:


obj=model_fitting(256,256)
# J = obj.Gauss_Newton_optim()


# In[ ]:


obj.cal_ver_alb(obj.chi[0:100],obj.chi[100:179],obj.chi[179:279])


# In[ ]:


obj.Gauss_Newton_optim()

