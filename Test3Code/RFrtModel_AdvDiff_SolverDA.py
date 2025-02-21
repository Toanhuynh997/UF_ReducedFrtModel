import torch
import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags, block_diag, vstack, hstack, coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve

class Subdomain:
    def __init__(self):
        # Properties
        self.RectMesh = None
        self.Precond = None
        self.nt = None
        self.dt = None
        self.xa = None
        self.xb = None
        self.yc = None
        self.yd = None
        self.elem_loc2glob = None
        self.elem_glob2loc = None
        self.edge_loc2glob = None
        self.edge_glob2loc = None
        self.boundary_local_id = None  # boundary edges with local edge index
        self.boundary_global_id = None  # boundary edges with global edge index
        self.bd_local = None  # boundary edges with local boundary index [left right bottom top]
        self.bd_global = None  # boundary edges with global boundary index
        self.neumann_bd = None
        self.neumann_bd_transport = None
        self.if_local_id = None  # interface edges with local edge index
        self.if_local_bd = None  # interface edges with local boundary index
        self.if_global_id = None
        self.all_bc_Darcy = None
        self.all_bc = None
        self.p0 = None
        self.p_prev = None
        self.Wloc = None
        self.Wrhs = None
        self.Wlambda = None
        self.Wtotal = None
        self.proj_loc2if = None
        self.proj_if2loc = None
        self.invKele = None
        self.invDele = None
        self.Dele = None
        self.velo = None
        self.poros = None
        self.diff = None
        self.hydraulic = None
        self.ventcell_bd = None
        self.ventcell_bd_transport = None
        self.n_ori_out = None
        
class RectMesh:
    def __init__(self):
        self.NumEgsHor = None
        self.NumEgsVer = None
        self.NumEgs = None
        self.NumEms = None
        self.NumDofus = None
        self.xgrid = None
        self.ygrid = None
#         self.EqnBC_Glob = None
        self.edge2elem = None
        self.edgeDir = None
        self.NumBdEgs = None
        self.boundaryPos = None
        self.bdLeft = None
        self.bdRight = None
        self.bdBot = None
        self.bdTop = None
        self.xs = 0
        self.xe = 0
        self.ys = 0
        self.ye = 0        
        self.bdLbc = None
        self.bdRbc = None
        self.bdBbc = None
        self.bdTbc = None
        self.bdLbc_transport = None
        self.bdRbc_transport = None
        self.bdBbc_transport = None
        self.bdTbc_transport = None
        self.boundary2edge = None
        self.edge2boundary = None
        self.edge2boundaryforMat = None
        self.edge2elem = None
        self.elem_xcoor = None
        self.elem_ycoor = None
        self.elem_hx = None
        self.elem_hy = None
        self.elem_center = None
        self.elem2edge = None
        self.elem2dofu = None
        self.dofu2elem = None
        self.dofu2edge = None
        self.dofu2bd = None
        self.bd2dofu = None
        self.bd2dofu_nofrac = None
        self.NumIntEgs = None
        self.IntEdge2dofu = None
        self.IntEdge2Edge = None
        self.Edge2IntEdge = None
        self.Edge2IntEdgeforMat = None
        self.Stiffmat_Darcy = None
        self.n_ori = None
        self.n_ori_out = None
        self.M_transport = None
        
        
    def RectMesh_Gen_AdvDiff(self, xa, xb, nx, yc, yd, ny, device):
        self.NumEgsHor = nx * (ny + 1)
        self.NumEgsVer = (nx + 1) * ny
        self.NumEgs = self.NumEgsHor+self.NumEgsVer
        self.NumEms = nx*ny
        self.NumDofus = 4*self.NumEms

        xgrid = torch.linspace(xa, xb, nx + 1)
        self.xgrid = xgrid
        ygrid = torch.linspace(yc, yd, ny + 1)
        self.ygrid = ygrid

        # Secondary mesh info on edge-vs-elements
        self.edge2elem = torch.zeros((self.NumEgs, 2), dtype=torch.int64, device=device)
        self.edgeDir = torch.zeros(self.NumEgs, dtype=torch.int64, device=device)

        # Boundary info
        self.NumBdEgs = 2 * (nx + ny)
        NumBdEgs = self.NumBdEgs

        self.boundaryPos = torch.zeros(NumBdEgs, dtype=torch.int64, device=device)
        self.bdLeft = torch.zeros((ny, 3), dtype=torch.float64, device=device)
        self.bdRight = torch.zeros((ny, 3), dtype=torch.float64, device=device)
        self.bdBot = torch.zeros((nx, 3), dtype=torch.float64, device=device)
        self.bdTop = torch.zeros((nx, 3), dtype=torch.float64, device=device)

        # Boundary indices
        self.bdLeft[:, 0] = torch.round(torch.arange(0, ny))  # 1-based -> 0-based
        self.bdRight[:, 0] = torch.arange(ny, 2*ny)
        self.bdBot[:, 0] = torch.arange(2*ny, 2*ny+nx)
        self.bdTop[:, 0] = torch.arange(2*ny+nx, 2*ny+2*nx)

        # Boundary coordinates
        self.bdLeft[:, 1] = ygrid[:-1]  # First column
        self.bdLeft[:, 2] = ygrid[1:]   # Second column
        self.bdRight[:, 1] = ygrid[:-1]
        self.bdRight[:, 2] = ygrid[1:]
        self.bdBot[:, 1] = xgrid[:-1]
        self.bdBot[:, 2] = xgrid[1:]
        self.bdTop[:, 1] = xgrid[:-1]
        self.bdTop[:, 2] = xgrid[1:]

        # Boundary conditions: Darcy
        self.bdLbc = torch.zeros((ny, 2), dtype=torch.float64, device=device)
        self.bdRbc = torch.zeros((ny, 2), dtype=torch.float64, device=device)
        self.bdBbc = torch.zeros((nx, 2), dtype=torch.float64, device=device)
        self.bdTbc = torch.zeros((nx, 2), dtype=torch.float64, device=device)
        
        # Boundary conditions: Transport
        self.bdLbc_transport = torch.zeros((ny, 2), dtype=torch.float64, device=device)
        self.bdRbc_transport = torch.zeros((ny, 2), dtype=torch.float64, device=device)
        self.bdBbc_transport = torch.zeros((nx, 2), dtype=torch.float64, device=device)
        self.bdTbc_transport = torch.zeros((nx, 2), dtype=torch.float64, device=device)
        
        # More mesh info 
        vert_edge = torch.arange(0, (nx + 1)*ny, dtype=torch.int64).reshape(ny, nx + 1)  # 1-based -> 0-based
        horz_edge = self.NumEgsVer + torch.arange(0, nx*(ny + 1), dtype=torch.int64).reshape(ny+1, nx)  # 1-based -> 0-based

        # Boundary to edge mapping
        self.boundary2edge = torch.cat([
            vert_edge[:, 0],  # Left boundary
            vert_edge[:, -1],  # Right boundary
            horz_edge[0, :],  # Bottom boundary
            horz_edge[-1, :]  # Top boundary
        ]).flatten()

        # Edge to boundary mapping
        self.edge2boundary = torch.zeros(self.NumEgs, dtype=torch.int64, device=device)
        self.edge2boundaryforMat = torch.zeros(self.NumEgs, dtype=torch.int64, device=device)
        for l in range(self.NumBdEgs):
            self.edge2boundary[self.boundary2edge[l]] = l 
            self.edge2boundaryforMat[self.boundary2edge[l]] = l+1

        # Vertical edges
        for j in range(ny):  # 1-based -> 0-based
            for i in range(nx + 1):  # 1-based -> 0-based
                k = j * (nx + 1) + i
                ie = i + j * nx
                self.edgeDir[k] = 2
                if i == 0:
                    self.edge2elem[k, :2] = torch.tensor([ie, -1], dtype=torch.int64, device=device)
                elif i == nx:
                    self.edge2elem[k, :2] = torch.tensor([ie - 1, -1], dtype=torch.int64, device=device)
                else:
                    self.edge2elem[k, :2] = torch.tensor([ie - 1, ie], dtype=torch.int64, device=device)

        # Horizontal edges   
        for j in range(ny + 1):  # 1-based -> 0-based
            for i in range(nx):  # 1-based -> 0-based
                k = self.NumEgsVer + j * nx + i
                self.edgeDir[k] = -2
                if j == 0:
                    ie = i + j * nx
                    self.edge2elem[k, :2] = torch.tensor([ie, -1], dtype=torch.int64, device=device)
                elif j == ny:
                    ie = i + (ny - 1) * nx
                    self.edge2elem[k, :2] = torch.tensor([ie, -1], dtype=torch.int64, device=device)
                else:
                    ie = i + j * nx
                    ie_nei = i + (j - 1) * nx
                    self.edge2elem[k, :2] = torch.tensor([ie_nei, ie], dtype=torch.int64, device=device)

        # Initialize element information
        self.elem_xcoor = torch.zeros((self.NumEms, 2), dtype=torch.float64, device=device)
        self.elem_ycoor = torch.zeros((self.NumEms, 2), dtype=torch.float64, device=device)
        self.elem_hx = torch.zeros(self.NumEms, dtype=torch.float64, device=device)
        self.elem_hy = torch.zeros(self.NumEms, dtype=torch.float64, device=device)
        self.elem_center = torch.zeros((self.NumEms, 2), dtype=torch.float64, device=device)

        # Element-to-edge and element-to-DOF mappings
        self.elem2edge = torch.zeros((self.NumEms, 4), dtype=torch.int64, device=device)
        self.elem2dofu = torch.zeros((self.NumEms, 4), dtype=torch.int64, device=device)

        # DOF mappings
        self.dofu2elem = torch.zeros(self.NumDofus, dtype=torch.int64, device=device)
        self.dofu2edge = torch.zeros(self.NumDofus, dtype=torch.int64, device=device)
        self.dofu2bd = torch.zeros(self.NumDofus, dtype=torch.int64, device=device)

        # Boundary to DOF mapping
        self.bd2dofu = torch.zeros(self.NumBdEgs, dtype=torch.int64, device=device)

        # More info v2
        for j in range(ny):  # 1-based -> 0-based
            for i in range(nx):  # 1-based -> 0-based
                k = i + j * nx

                # Element k: [xa, xb] x [yc, yd]
                self.elem_xcoor[k, 0] = xgrid[i]  # xa
                self.elem_xcoor[k, 1] = xgrid[i + 1]  # xb
                self.elem_hx[k] = self.elem_xcoor[k, 1] - self.elem_xcoor[k, 0]
                self.elem_ycoor[k, 0] = ygrid[j]  # yc
                self.elem_ycoor[k, 1] = ygrid[j + 1]  # yd
                self.elem_hy[k] = self.elem_ycoor[k, 1] - self.elem_ycoor[k, 0]
                self.elem_center[k, 0] = (self.elem_xcoor[k, 1] + self.elem_xcoor[k, 0]) / 2  # xmid
                self.elem_center[k, 1] = (self.elem_ycoor[k, 1] + self.elem_ycoor[k, 0]) / 2  # ymid

                # Edge indices for element k
                edgeV = i + j * (nx + 1)
                edgeH = self.NumEgsVer + j * nx + i
                self.elem2edge[k, 0] = edgeV
                self.elem2edge[k, 1] = edgeV + 1
                self.elem2edge[k, 2] = edgeH
                self.elem2edge[k, 3] = edgeH + nx

                il = k * 4
                self.elem2dofu[k, 0] = il
                self.elem2dofu[k, 1] = il + 1
                self.elem2dofu[k, 2] = il + 2
                self.elem2dofu[k, 3] = il + 3

                # Mapping DOF to element and edge
                self.dofu2elem[il:il + 4] = k
                self.dofu2edge[il] = edgeV
                self.dofu2edge[il + 1] = edgeV + 1
                self.dofu2edge[il + 2] = edgeH
                self.dofu2edge[il + 3] = edgeH + nx

                # Boundary DOF mapping          
                if self.edge2elem[edgeV, 1] == -1:
                    bd_id = self.edge2boundary[edgeV]
                    self.dofu2bd[il] = bd_id
                    self.bd2dofu[bd_id] = il


                if self.edge2elem[edgeV + 1, 1] == -1:
                    bd_id = self.edge2boundary[edgeV + 1]
                    self.dofu2bd[il + 1] = bd_id
                    self.bd2dofu[bd_id] = il + 1


                if self.edge2elem[edgeH, 1] == -1:
                    bd_id = self.edge2boundary[edgeH]
                    self.dofu2bd[il + 2] = bd_id
                    self.bd2dofu[bd_id] = il + 2


                if self.edge2elem[edgeH + nx, 1] == -1:
                    bd_id = self.edge2boundary[edgeH + nx]
                    self.dofu2bd[il + 3] = bd_id
                    self.bd2dofu[bd_id] = il + 3

        self.NumIntEgs = self.NumEgs - self.NumBdEgs
        self.IntEdge2dofu = torch.zeros((self.NumIntEgs, 3), dtype=torch.int64)  # Internal edge DOFs
        self.IntEdge2Edge = torch.zeros((self.NumEgs,), dtype=torch.int64)      # Edge-to-IntEdge mapping
        self.Edge2IntEdge = torch.zeros((self.NumEgs,), dtype=torch.int64)      # IntEdge-to-Edge mapping
        self.Edge2IntEdgeforMat = torch.zeros((self.NumEgs,), dtype=torch.int64)
        inte_count = 0

        for k in range(self.NumEgs):  # Loop over all edges
            if self.edgeDir[k] == 2 and self.edge2elem[k, 1] != -1:  # Vertical, not boundary
                inte_count += 1
                self.IntEdge2Edge[inte_count - 1] = k 
                self.Edge2IntEdge[k] = inte_count-1
                self.Edge2IntEdgeforMat[k] = inte_count-1
                left_elem = self.edge2elem[k, 0] 
                right_elem = self.edge2elem[k, 1]
                self.IntEdge2dofu[inte_count - 1, 1] = self.elem2dofu[left_elem, 1]  # Left element DOF
                self.IntEdge2dofu[inte_count - 1, 2] = self.elem2dofu[right_elem, 0]  # Right element DOF

            if self.edgeDir[k] == -2 and self.edge2elem[k, 1] != -1:  # Horizontal, not boundary
                inte_count += 1
                self.IntEdge2Edge[inte_count - 1] = k 
                self.Edge2IntEdge[k] = inte_count-1
                self.Edge2IntEdgeforMat[k] = inte_count
                bottom_elem = self.edge2elem[k, 0] 
                top_elem = self.edge2elem[k, 1] 
                self.IntEdge2dofu[inte_count - 1, 1] = self.elem2dofu[bottom_elem, 3]  # Bottom element DOF
                self.IntEdge2dofu[inte_count - 1, 2] = self.elem2dofu[top_elem, 2]  # Top element DOF      
        return self
    
        

def stripdecomposition_advdiff(nx, ny, NPX, NPY, invKele, invDele, poros, diff, hydraulic):
    # Define domain boundaries
    xa, xb, yc, yd = 0, 2, 0, 1

    gnx, gny = nx * NPX, ny * NPY
    gNx, gNy = gnx + 1, gny + 1
    Nx, Ny = nx + 1, ny + 1

    count_gele = 0  # Global element index of left-bottom corner of each subdomain
    count_gvert = 0
    count_ghorz = gNx * gny

    # Subdomain boundary indices: left, right, bottom, top
    vert_edge = torch.arange(Nx * ny).reshape(ny, Nx)  # ny x Nx matrix
    horz_edge = Nx * ny + torch.arange(nx * Ny).reshape(Ny, nx)  # Ny x nx matrix
    left = vert_edge[:, 0]  # ny x 1
    right = vert_edge[:, -1]
    bottom = horz_edge[0, :]  # nx x 1
    top = horz_edge[-1, :]
    # Global element indices
    # Um = torch.arange(nx * ny).reshape(Nx, Ny).T

    # Subdomains
    nSub = NPX * NPY
    Xgrid = torch.linspace(xa, xb, NPX + 1)
    Ygrid = torch.linspace(yc, yd, NPY + 1)

    subdomain = [None] * nSub  # Placeholder for subdomain objects
    omega = type('domain', (object,), {})()  # Dynamically define a domain class

    count_if = 0  # Interface edge numbering

    if NPY == 1 and NPX > 1:  # Vertical decomposition
        omega.if_size = ny * (NPX - 1)
        omega.if_local_to_global = torch.zeros(omega.if_size, dtype=torch.int64)
        omega.if_global_to_local = torch.zeros(gNx * gny + gNy * gnx, dtype=torch.int64)

        for k in range(NPX):
            # Create a new subdomain object
            subdomain[k] = Subdomain()
            subdomain[k].xa = Xgrid[k]
            subdomain[k].xb = Xgrid[k + 1]
            subdomain[k].yc = Ygrid[0]
            subdomain[k].yd = Ygrid[1]
            subdomain[k].elem_loc2glob = torch.zeros(nx * ny, dtype=torch.int64)  # Element indices
            subdomain[k].elem_glob2loc = torch.zeros(gnx * gny, dtype=torch.int64)

            subdomain[k].edge_loc2glob = torch.zeros(Nx * ny + nx * Ny, dtype=torch.int64)  # Edge indices
            subdomain[k].edge_glob2loc = torch.zeros(gNx * gny + gnx * gNy, dtype=torch.int64)

            count_lele = 0
            count_lvert = 0
            count_lhorz = Nx * ny

            for j1 in range(ny):
                # Elements
                istart = count_gele + j1 * gnx
                subdomain[k].elem_loc2glob[count_lele:count_lele + nx] = torch.arange(istart, istart + nx)
                subdomain[k].elem_glob2loc[istart:istart + nx] = torch.arange(count_lele, count_lele + nx)
                count_lele += nx

                # Vertical edges
                istart_vert = count_gvert + j1 * gNx
                subdomain[k].edge_loc2glob[count_lvert:count_lvert + Nx] = torch.arange(istart_vert, istart_vert + Nx)
                subdomain[k].edge_glob2loc[istart_vert:istart_vert + Nx] = torch.arange(count_lvert, count_lvert + Nx)
                count_lvert += Nx

                # Horizontal edges
                istart_horz = count_ghorz + j1 * gnx
                subdomain[k].edge_loc2glob[count_lhorz:count_lhorz + nx] = torch.arange(istart_horz, istart_horz + nx)
                subdomain[k].edge_glob2loc[istart_horz:istart_horz + nx] = torch.arange(count_lhorz, count_lhorz + nx)
                count_lhorz += nx
                
            istart_horz = count_ghorz + ny * gnx
            subdomain[k].edge_loc2glob[count_lhorz:count_lhorz + nx] = torch.arange(istart_horz, istart_horz + nx)
            subdomain[k].edge_glob2loc[istart_horz:istart_horz + nx] = torch.arange(count_lhorz, count_lhorz + nx)

            count_gele += nx
            count_gvert += nx
            count_ghorz += nx

            if k == 0:  # First subdomain (left)
                subdomain[k].boundary_local_id = torch.zeros(2 * nx + ny, dtype=torch.int64)
                subdomain[k].boundary_global_id = torch.zeros(2 * nx + ny, dtype=torch.int64)

                local_bd = torch.cat([left, bottom, top])  # Boundary with edge index
                subdomain[k].boundary_local_id = local_bd
                subdomain[k].boundary_global_id = subdomain[k].edge_loc2glob[local_bd]

                subdomain[k].if_local_id = torch.zeros(ny, dtype=torch.int64)
                subdomain[k].if_global_id = torch.zeros(ny, dtype=torch.int64)

                subdomain[k].if_local_id = right
                subdomain[k].if_global_id = subdomain[k].edge_loc2glob[right]

                omega.if_local_to_global[count_if:count_if + ny] = subdomain[k].if_global_id
                omega.if_global_to_local[subdomain[k].if_global_id] = torch.arange(count_if, count_if + ny)
                count_if += ny

            elif k == NPX - 1:  # Last subdomain (right)
                subdomain[k].boundary_local_id = torch.zeros(2 * nx + ny, dtype=torch.int64)
                subdomain[k].boundary_global_id = torch.zeros(2 * nx + ny, dtype=torch.int64)

                local_bd = torch.cat([right, bottom, top])
                subdomain[k].boundary_local_id = local_bd
                subdomain[k].boundary_global_id = subdomain[k].edge_loc2glob[local_bd]

                subdomain[k].if_local_id = torch.zeros(ny, dtype=torch.int64)
                subdomain[k].if_global_id = torch.zeros(ny, dtype=torch.int64)

                subdomain[k].if_local_id = left
                subdomain[k].if_global_id = subdomain[k].edge_loc2glob[left]

            else:  # Internal subdomains
                subdomain[k].boundary_local_id = torch.zeros(2 * nx, dtype=torch.int64)
                subdomain[k].boundary_global_id = torch.zeros(2 * nx, dtype=torch.int64)

                local_bd = torch.cat([bottom, top])
                subdomain[k].boundary_local_id = local_bd
                subdomain[k].boundary_global_id = subdomain[k].edge_loc2glob[local_bd]

                subdomain[k].if_local_id = torch.zeros(2 * ny, dtype=torch.int64)
                subdomain[k].if_global_id = torch.zeros(2 * ny, dtype=torch.int64)

                # Right interface first
                subdomain[k].if_local_id[:ny] = right
                subdomain[k].if_global_id[:ny] = subdomain[k].edge_loc2glob[right]

                omega.if_local_to_global[count_if:count_if + ny] = subdomain[k].if_global_id[:ny]
                omega.if_global_to_local[subdomain[k].if_global_id[:ny]] = torch.arange(count_if, count_if + ny)
                count_if += ny

                # Then left interface
                subdomain[k].if_local_id[ny:2 * ny] = left
                subdomain[k].if_global_id[ny:2 * ny] = subdomain[k].edge_loc2glob[left]

            subdomain[k].invKele = torch.zeros(nx * ny, dtype=torch.float64)
            subdomain[k].invKele = invKele[subdomain[k].elem_loc2glob]

            subdomain[k].invDele = torch.zeros(nx * ny, dtype=torch.float64)
            subdomain[k].invDele = invDele[subdomain[k].elem_loc2glob]

            subdomain[k].poros = poros[k]
            subdomain[k].diff = diff[k]
            subdomain[k].hydraulic = hydraulic[k]
            
    return subdomain
        
        
def GenerateSubdom_AdvDiff(nx, ny, xa, xb, yc, yd, delta, NPX, NPY, nt, t0, T, device):
    # Global parameters
    n_ori_out = [None] * 2
       
    n_ori = [None]*2 # Replace with actual n_ori values
#     global xa, xb, yc, yd, delta
#     xa, xb = 0, 2
#     yc, yd = 0, 1
    
    len_edge_frac = (yd - yc) / ny
    dy = len_edge_frac
    
    # Diffusion and hydraulic constants
    diff_1, diff_2 = 3.15e-4, 3.15e-4 #1, 1
    
    diff_gamma = 9.92e-3 # 31.5
    diff = torch.tensor([diff_1, diff_2], dtype=torch.float64)
    
    hydraulic_1, hydraulic_2 = 9.92e-6, 9.92e-6 #3.15e-02
    hydraulic_gamma = 3.15e-5 # 0.1
    hydraulic = torch.tensor([hydraulic_1, hydraulic_2], dtype=torch.float64)

    delta = 0.1
    hydraulic1d = hydraulic_gamma * delta
    nSub = NPX * NPY
    nt_all = [nt, nt]
    
    # Initial condition grids
    xgrid = [None] * 2
    ygrid = [None] * nSub

    xgrid[0] = torch.linspace(xa, (xa + xb) / 2, nx + 1)
    xgrid[1] = torch.linspace((xa + xb) / 2, xb, nx + 1)
    for k in range(nSub):
        ygrid[k] = torch.linspace(yc, yd, ny + 1)

    # Compute midpoints for each grid
    xmid = [None] * 2
    ymid = [None] * nSub
    for k in range(nSub):
        xmid[k] = (xgrid[k][:-1] + xgrid[k][1:]) / 2
        ymid[k] = (ygrid[k][:-1] + ygrid[k][1:]) / 2

    Xmid = [None] * 2
    Ymid = [None] * 2
    for k in range(2):
        Xmid[k], Ymid[k] = torch.meshgrid(xmid[k], ymid[k], indexing='ij')

    # Global mesh
    gnx = nx * NPX
    gny = ny * NPY

    xgrid_all = torch.linspace(xa, xb, gnx + 1)
    ygrid_all = torch.linspace(yc, yd, gny + 1)

    xmid_all = (xgrid_all[:-1] + xgrid_all[1:]) / 2
    ymid_all = (ygrid_all[:-1] + ygrid_all[1:]) / 2

    Xmid_all, Ymid_all = torch.meshgrid(xmid_all, ymid_all, indexing='ij')
    
    # Assuming `MeshGen_new` is a function that generates a mesh (define separately)
    gRectMesh = RectMesh()
    gRectMesh.RectMesh_Gen_AdvDiff(xa, xb, gnx, yc, yd, gny, device)

    # Generate diffusion per element
    invKele_m = torch.zeros((gny, gnx), dtype = torch.float64)
    invKele_m[:, :gnx // 2] = 1 / hydraulic_1
    invKele_m[:, gnx // 2:] = 1 / hydraulic_2
    invKele = invKele_m.T.flatten()

    invDele_m = torch.zeros((gny, gnx), dtype = torch.float64)
    invDele_m[:, :gnx // 2] = 1 / diff_1
    invDele_m[:, gnx // 2:] = 1 / diff_2
    invDele = invDele_m.T.flatten()

    # Porosity
    poros = [0.05, 0.05, 0.1]  # Including fracture porosity
    # Boundary condition setup
    gRectMesh.bdLbc = torch.zeros((gny, 2), dtype = torch.float64)
    gRectMesh.bdRbc = torch.zeros((gny, 2), dtype = torch.float64)
    gRectMesh.bdBbc = torch.zeros((gnx, 2), dtype = torch.float64)
    gRectMesh.bdTbc = torch.zeros((gnx, 2), dtype = torch.float64)

    gRectMesh.bdLbc[:, 0] = 2  # Neumann
    gRectMesh.bdRbc[:, 0] = 2  # Neumann
    gRectMesh.bdBbc[:, 0] = 1  # Dirichlet
    gRectMesh.bdTbc[:, 0] = 1  # Dirichlet

    # Bottom boundary pressure profile
    Pres_frt_bot = 3
    Pres_bot = [torch.zeros(nx, dtype=torch.float64), torch.zeros(nx, dtype=torch.float64)]

    for i in range(nx):
        Pres_bot[0][i] = Pres_frt_bot + (nx + 1 - (i + 1)) * 0.05
        Pres_bot[1][i] = Pres_frt_bot + (i + 1) * 0.05

    gRectMesh.bdBbc[:gnx // 2, 1] = Pres_bot[0]
    gRectMesh.bdBbc[gnx // 2:, 1] = Pres_bot[1]

    all_bc_Darcy = torch.vstack([
        gRectMesh.bdLbc,
        gRectMesh.bdRbc,
        gRectMesh.bdBbc,
        gRectMesh.bdTbc
    ])
#     print(all_bc_Darcy)
    gRectMesh.AuBd = torch.zeros(gRectMesh.NumBdEgs, dtype=torch.float64)

    neumann_bd = [i for i, bc in enumerate(all_bc_Darcy) if bc[0] == 2]

    # Subdomain decomposition
    subdomain = stripdecomposition_advdiff(nx, ny, NPX, NPY, invKele, invDele, poros, diff, hydraulic)
    
    subdomain[0].Dele = diff_1
    subdomain[1].Dele = diff_2
    
    tgrid_glob = torch.linspace(t0, T, nt + 1)
    for k in range(nSub):
        # Time grids and projections
        subdomain[k].nt = nt_all[k]
        tgrid_loc = torch.linspace(t0, T, subdomain[k].nt + 1)
        subdomain[k].dt = (T - t0) / subdomain[k].nt

        # Local mesh
        subdomain[k].RectMesh = RectMesh()
        subdomain[k].RectMesh.RectMesh_Gen_AdvDiff(
            subdomain[k].xa, subdomain[k].xb, nx,
            subdomain[k].yc, subdomain[k].yd, ny, device
        )

        subdomain[k].if_local_bd = subdomain[k].RectMesh.edge2boundary[subdomain[k].if_local_id]

        # Read boundary condition type
        subdomain[k].all_bc_Darcy = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 2), dtype=torch.float64)
        subdomain[k].bd_local = subdomain[k].RectMesh.edge2boundary[subdomain[k].boundary_local_id]
        subdomain[k].bd_global = gRectMesh.edge2boundary[subdomain[k].boundary_global_id]
              
        subdomain[k].all_bc_Darcy[subdomain[k].bd_local, 0] = all_bc_Darcy[subdomain[k].bd_global, 0]
        subdomain[k].all_bc_Darcy[subdomain[k].if_local_bd, 0] = 4  # Coupled with interface
        subdomain[k].neumann_bd = [i for i, bc in enumerate(subdomain[k].all_bc_Darcy) if bc[0] == 2]

        subdomain[k].RectMesh.AuBd = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 4), dtype=torch.float64)
        subdomain[k].RectMesh.Bdtheta2dofu = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 4), dtype=torch.float64)
        subdomain[k].ventcell_bd = []  # Initialize as an empty list

        # Iterate through subdomains and their all_bc_Darcy to find Ventcell BCs
        n_ori[k]= subdomain[k].RectMesh.NumDofus+subdomain[k].RectMesh.NumEms+subdomain[k].RectMesh.NumIntEgs

        n_ori_out[k] = n_ori[k] + subdomain[k].RectMesh.NumBdEgs - ny
        
        subdomain[k].RectMesh.n_ori = n_ori[k]
        subdomain[k].RectMesh.n_ori_out = n_ori_out[k]
        
        for i in range(len(subdomain[k].all_bc_Darcy)):
            if subdomain[k].all_bc_Darcy[i, 0] == 4:  # Check for Ventcell BCs
                subdomain[k].ventcell_bd.append(i)
                
        # Stiffness matrix and Darcy solver
        LMat = stiffmat_SubDom(
            subdomain[k].RectMesh, subdomain[k].neumann_bd,
            subdomain[k].invKele, nx, ny, k
        )
        subdomain[k].RectMesh.Stiffmat_Darcy = LMat
        
    subdomain[0].RectMesh.bdLbc = gRectMesh.bdLbc
    subdomain[1].RectMesh.bdRbc = gRectMesh.bdRbc

    subdomain[0].RectMesh.bdTbc[:, 0] = 1
    subdomain[0].RectMesh.bdTbc[:, 1] = gRectMesh.bdTbc[:gnx // 2, 1]

    subdomain[1].RectMesh.bdTbc[:, 0] = 1
    subdomain[1].RectMesh.bdTbc[:, 1] = gRectMesh.bdTbc[gnx // 2:, 1]

    subdomain[0].RectMesh.bdBbc[:, 0] = 1
    subdomain[0].RectMesh.bdBbc[:, 1] = gRectMesh.bdBbc[:gnx // 2, 1]

    subdomain[1].RectMesh.bdBbc[:, 0] = 1
    subdomain[1].RectMesh.bdBbc[:, 1] = gRectMesh.bdBbc[gnx // 2:, 1]  
    return subdomain, hydraulic1d, len_edge_frac, diff_gamma, poros

def GenerateSubdom_AdvDiff_v2(nx, ny, xa, xb, yc, yd, delta, NPX, NPY, nt, t0, T, d1, d2, dfrt, k1, k2, kfrt, device):
    # Global parameters
    n_ori_out = [None] * 2
       
    n_ori = [None]*2 # Replace with actual n_ori values
#     global xa, xb, yc, yd, delta
#     xa, xb = 0, 2
#     yc, yd = 0, 1
    
    len_edge_frac = (yd - yc) / ny
    dy = len_edge_frac
    
    # Diffusion and hydraulic constants
    diff_1, diff_2 = d1, d2
    
    diff_gamma = dfrt
    diff = torch.tensor([diff_1, diff_2], dtype=torch.float64)

    hydraulic_1, hydraulic_2 = k1, k2
    hydraulic_gamma = kfrt
    hydraulic = torch.tensor([hydraulic_1, hydraulic_2], dtype=torch.float64)

    delta = 0.1
    hydraulic1d = hydraulic_gamma * delta
    nSub = NPX * NPY
    nt_all = [nt, nt]
    
    # Initial condition grids
    xgrid = [None] * 2
    ygrid = [None] * nSub

    xgrid[0] = torch.linspace(xa, (xa + xb) / 2, nx + 1)
    xgrid[1] = torch.linspace((xa + xb) / 2, xb, nx + 1)
    for k in range(nSub):
        ygrid[k] = torch.linspace(yc, yd, ny + 1)

    # Compute midpoints for each grid
    xmid = [None] * 2
    ymid = [None] * nSub
    for k in range(nSub):
        xmid[k] = (xgrid[k][:-1] + xgrid[k][1:]) / 2
        ymid[k] = (ygrid[k][:-1] + ygrid[k][1:]) / 2

    Xmid = [None] * 2
    Ymid = [None] * 2
    for k in range(2):
        Xmid[k], Ymid[k] = torch.meshgrid(xmid[k], ymid[k], indexing='ij')

    # Global mesh
    gnx = nx * NPX
    gny = ny * NPY

    xgrid_all = torch.linspace(xa, xb, gnx + 1)
    ygrid_all = torch.linspace(yc, yd, gny + 1)

    xmid_all = (xgrid_all[:-1] + xgrid_all[1:]) / 2
    ymid_all = (ygrid_all[:-1] + ygrid_all[1:]) / 2

    Xmid_all, Ymid_all = torch.meshgrid(xmid_all, ymid_all, indexing='ij')
    
    # Assuming `MeshGen_new` is a function that generates a mesh (define separately)
    gRectMesh = RectMesh()
    gRectMesh.RectMesh_Gen_AdvDiff(xa, xb, gnx, yc, yd, gny, device)

    # Generate diffusion per element
    invKele_m = torch.zeros((gny, gnx), dtype = torch.float64)
    invKele_m[:, :gnx // 2] = 1 / hydraulic_1
    invKele_m[:, gnx // 2:] = 1 / hydraulic_2
    invKele = invKele_m.T.flatten()

    invDele_m = torch.zeros((gny, gnx), dtype = torch.float64)
    invDele_m[:, :gnx // 2] = 1 / diff_1
    invDele_m[:, gnx // 2:] = 1 / diff_2
    invDele = invDele_m.T.flatten()

    # Porosity
    poros = [0.05, 0.05, 0.1]  # Including fracture porosity
    # Boundary condition setup
    gRectMesh.bdLbc = torch.zeros((gny, 2), dtype = torch.float64)
    gRectMesh.bdRbc = torch.zeros((gny, 2), dtype = torch.float64)
    gRectMesh.bdBbc = torch.zeros((gnx, 2), dtype = torch.float64)
    gRectMesh.bdTbc = torch.zeros((gnx, 2), dtype = torch.float64)

    gRectMesh.bdLbc[:, 0] = 2  # Neumann
    gRectMesh.bdRbc[:, 0] = 2  # Neumann
    gRectMesh.bdBbc[:, 0] = 1  # Dirichlet
    gRectMesh.bdTbc[:, 0] = 1  # Dirichlet

    # Bottom boundary pressure profile
    Pres_frt_bot = 3
    Pres_bot = [torch.zeros(nx, dtype=torch.float64), torch.zeros(nx, dtype=torch.float64)]

    for i in range(nx):
        Pres_bot[0][i] = Pres_frt_bot + (nx + 1 - (i + 1)) * 0.05
        Pres_bot[1][i] = Pres_frt_bot + (i + 1) * 0.05

    gRectMesh.bdBbc[:gnx // 2, 1] = Pres_bot[0]
    gRectMesh.bdBbc[gnx // 2:, 1] = Pres_bot[1]

    all_bc_Darcy = torch.vstack([
        gRectMesh.bdLbc,
        gRectMesh.bdRbc,
        gRectMesh.bdBbc,
        gRectMesh.bdTbc
    ])
#     print(all_bc_Darcy)
    gRectMesh.AuBd = torch.zeros(gRectMesh.NumBdEgs, dtype=torch.float64)

    neumann_bd = [i for i, bc in enumerate(all_bc_Darcy) if bc[0] == 2]

    # Subdomain decomposition
    subdomain = stripdecomposition_advdiff(nx, ny, NPX, NPY, invKele, invDele, poros, diff, hydraulic)
    
    subdomain[0].Dele = diff_1
    subdomain[1].Dele = diff_2
    
    tgrid_glob = torch.linspace(t0, T, nt + 1)
    for k in range(nSub):
        # Time grids and projections
        subdomain[k].nt = nt_all[k]
        tgrid_loc = torch.linspace(t0, T, subdomain[k].nt + 1)
        subdomain[k].dt = (T - t0) / subdomain[k].nt

        # Local mesh
        subdomain[k].RectMesh = RectMesh()
        subdomain[k].RectMesh.RectMesh_Gen_AdvDiff(
            subdomain[k].xa, subdomain[k].xb, nx,
            subdomain[k].yc, subdomain[k].yd, ny, device
        )

        subdomain[k].if_local_bd = subdomain[k].RectMesh.edge2boundary[subdomain[k].if_local_id]

        # Read boundary condition type
        subdomain[k].all_bc_Darcy = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 2), dtype=torch.float64)
        subdomain[k].bd_local = subdomain[k].RectMesh.edge2boundary[subdomain[k].boundary_local_id]
        subdomain[k].bd_global = gRectMesh.edge2boundary[subdomain[k].boundary_global_id]
              
        subdomain[k].all_bc_Darcy[subdomain[k].bd_local, 0] = all_bc_Darcy[subdomain[k].bd_global, 0]
        subdomain[k].all_bc_Darcy[subdomain[k].if_local_bd, 0] = 4  # Coupled with interface

        subdomain[k].neumann_bd = [i for i, bc in enumerate(subdomain[k].all_bc_Darcy) if bc[0] == 2]

        subdomain[k].RectMesh.AuBd = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 4), dtype=torch.float64)
        subdomain[k].RectMesh.Bdtheta2dofu = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 4), dtype=torch.float64)
        subdomain[k].ventcell_bd = []  # Initialize as an empty list

        # Iterate through subdomains and their all_bc_Darcy to find Ventcell BCs
        n_ori[k]= subdomain[k].RectMesh.NumDofus+subdomain[k].RectMesh.NumEms+subdomain[k].RectMesh.NumIntEgs

        n_ori_out[k] = n_ori[k] + subdomain[k].RectMesh.NumBdEgs - ny
        
        subdomain[k].RectMesh.n_ori = n_ori[k]
        subdomain[k].RectMesh.n_ori_out = n_ori_out[k]
        
        for i in range(len(subdomain[k].all_bc_Darcy)):
            if subdomain[k].all_bc_Darcy[i, 0] == 4:  # Check for Ventcell BCs
                subdomain[k].ventcell_bd.append(i)
                
        # Stiffness matrix and Darcy solver
        LMat = stiffmat_SubDom(
            subdomain[k].RectMesh, subdomain[k].neumann_bd,
            subdomain[k].invKele, nx, ny, k
        )
        subdomain[k].RectMesh.Stiffmat_Darcy = LMat
        
    subdomain[0].RectMesh.bdLbc = gRectMesh.bdLbc
    subdomain[1].RectMesh.bdRbc = gRectMesh.bdRbc

    subdomain[0].RectMesh.bdTbc[:, 0] = 1
    subdomain[0].RectMesh.bdTbc[:, 1] = gRectMesh.bdTbc[:gnx // 2, 1]

    subdomain[1].RectMesh.bdTbc[:, 0] = 1
    subdomain[1].RectMesh.bdTbc[:, 1] = gRectMesh.bdTbc[gnx // 2:, 1]

    subdomain[0].RectMesh.bdBbc[:, 0] = 1
    subdomain[0].RectMesh.bdBbc[:, 1] = gRectMesh.bdBbc[:gnx // 2, 1]

    subdomain[1].RectMesh.bdBbc[:, 0] = 1
    subdomain[1].RectMesh.bdBbc[:, 1] = gRectMesh.bdBbc[gnx // 2:, 1]  
    return subdomain, hydraulic1d, len_edge_frac, diff_gamma, poros
    
def stiffmat_SubDom(RectMesh, neumann_bd, invPara, nx, ny, k):
    # Build block stiffness matrix
    # Matrix B
    Ivec = torch.arange(0, RectMesh.NumEms)
    Imat = Ivec.repeat(4, 1).T
    I = Imat.reshape(-1)
    J = torch.arange(0, RectMesh.NumDofus)
    val = -torch.ones(RectMesh.NumDofus)
#     print(I)
    B = torch.sparse_coo_tensor(
        indices=torch.vstack([I, J]),  # Stacking row and column indices
        values=val,
        size=(RectMesh.NumEms, RectMesh.NumDofus),
        dtype=torch.float64
    )
    
    # Matrix C
    Icvec = torch.arange(0, RectMesh.NumIntEgs)
    Icmat = Icvec.repeat(2, 1).T
    Ic = Icmat.reshape(-1)
#     print(Ic)
    Jcmat = RectMesh.IntEdge2dofu[:, 1:3]
    Jc = Jcmat.reshape(-1)
    valc = torch.ones(2 * RectMesh.NumIntEgs)
    
    C = torch.sparse_coo_tensor(
        indices=torch.vstack([Ic, Jc]),
        values=valc,
        size=(RectMesh.NumIntEgs, RectMesh.NumDofus),
        dtype=torch.float64
    )
    
    Ithe = RectMesh.bd2dofu  # Already a tensor
    Jthe = torch.arange(0, RectMesh.NumBdEgs, dtype=torch.int64)  # Column indices
    valthe = torch.ones(RectMesh.NumBdEgs, dtype=torch.float64)  # Values for sparse matrix
    # Create the sparse tensor Ctheta
    ### Matrix Ctheta
    v = torch.zeros(ny, 1, dtype=torch.int64);
    if k == 0:
        for i in range(1, ny+1):  # Loop from 1 to ny (MATLAB-style indexing)
            v[i-1] = i * nx  # MATLAB uses 1-based indexing; Python is 0-based
        u = RectMesh.elem2dofu[v-1, 1]  # Assuming elem2dofu is implemented for PyTorch
    elif k == 1:
        for i in range(1, ny+1):  # Loop from 1 to ny
            v[i-1] = (i - 1) * nx +1  # Corrected indexing for Python
        u = RectMesh.elem2dofu[v-1, 0]

    # Initialize r and remove elements in u
    r = RectMesh.bd2dofu.clone()  # Clone to avoid modifying the original tensor
    r = r[~torch.isin(r, u)]

    RectMesh.bd2dofu_nofrac = r

    # Create sparse matrix Ctheta
    Ithe = RectMesh.bd2dofu_nofrac  # Indices of rows
    Jthe = torch.arange(0, RectMesh.NumBdEgs - ny, dtype=torch.float64)  # Indices of columns
    valthe = torch.ones(RectMesh.NumBdEgs - ny, dtype=torch.float64)  # Values

    # Construct sparse tensor Ctheta
    indices = torch.stack([Ithe, Jthe])  # Combine row and column indices
    Ctheta = torch.sparse_coo_tensor(indices, valthe, 
                                     (RectMesh.NumDofus, RectMesh.NumBdEgs - ny),
                                     dtype=torch.float64)
   
    # Initialize sparse matrices and vectors
    A = torch.zeros((RectMesh.NumDofus, RectMesh.NumDofus), dtype=torch.float64)
    Au = torch.zeros((RectMesh.NumDofus, RectMesh.NumIntEgs), dtype=torch.float64)
    Au_c = torch.zeros(RectMesh.NumDofus, dtype=torch.float64)
    Atheta_nofrac = torch.zeros((RectMesh.NumDofus, RectMesh.NumBdEgs), dtype=torch.float64)
    
    for i in range(RectMesh.NumEms):
        str_ = RectMesh.elem2dofu[i, :]
        
        # Compute local stiffness matrix
        Ae = localA(invPara[i], RectMesh.elem_hx[i], RectMesh.elem_hy[i])

        A[torch.meshgrid(str_, str_, indexing='ij')] = Ae.clone()
        
    S=RectMesh.elem_hx*RectMesh.elem_hy   
    n_ori=RectMesh.NumDofus+RectMesh.NumEms+RectMesh.NumIntEgs
    
    L_top = torch.cat([
        torch.cat([A, B.to_dense().T, (C.to_dense().T), Ctheta.to_dense()], dim=1),
        torch.cat([
             B.to_dense(), torch.zeros(RectMesh.NumEms, RectMesh.NumEms, dtype=torch.float64),
            torch.zeros(RectMesh.NumEms, RectMesh.NumIntEgs+RectMesh.NumBdEgs-ny, dtype=torch.float64)
        ], dim=1),
        torch.cat([
            C.to_dense(),
            torch.zeros(RectMesh.NumIntEgs, RectMesh.NumEms+RectMesh.NumIntEgs+RectMesh.NumBdEgs-ny, dtype=torch.float64)
        ], dim=1)
    ], dim=0)

    L_bottom = torch.cat([
        torch.zeros(RectMesh.NumBdEgs-ny, n_ori, dtype=torch.float64),
        torch.eye(RectMesh.NumBdEgs-ny, dtype=torch.float64)
    ], dim=1)
    
    L = torch.cat([L_top, L_bottom], dim=0)
    
    nall = L.size(0)
    nneu = len(neumann_bd)

    for i in range(nneu):
        current_bd = neumann_bd[i]
        bddofu = RectMesh.bd2dofu[current_bd]  # Retrieve boundary degrees of freedom

        # Determine position 'posi' based on condition
        if current_bd <= ny-1:
            posi = n_ori + current_bd
        else:
            posi = n_ori + current_bd - ny
            
        # Update row in L
        L[posi, :] = 0  # Reset the entire row to zero (equivalent to sparse(1, nall))
        L[posi, bddofu] = 1  # Set the corresponding entry to 1
        
    L_dense = L.to_dense() if L.is_sparse else L
 
    return L_dense

def Assemble_StiffMat_Darcy(subdomain, ny, hydraulic1d, len_edge_frac):
    RectMesh = [None] * 2  # Placeholder for Sub_RectMesh
    for k0 in range(2):
        RectMesh[k0] = subdomain[k0].RectMesh
        
    nv = ny
    hy = RectMesh[0].elem_hy
    
    ### 1D Matrix
    Ivec1D = torch.arange(0, nv)
    Imat1D = Ivec1D.repeat(2, 1).T
    I1D = Imat1D.reshape(-1)
    J1D = torch.arange(0, 2*nv)
    val1D = -torch.ones(2 * nv, dtype=torch.float64)
    
    Bv1d = torch.sparse_coo_tensor(
        indices = torch.stack([I1D, J1D]),
        values = val1D,
        size = (nv, 2*nv),
        dtype = torch.float64
    )
    Bv1d = Bv1d.to_dense()
    Bv1d_t = torch.transpose(Bv1d, 0, 1)

    # build the 1D C matrix
    Ivec1D = torch.arange(0, nv-1)
    Imat1D = Ivec1D.repeat(2, 1).T
    I1D = Imat1D.reshape(-1)
    J1D = torch.arange(1, 2*nv-1)
    val1D = torch.ones(2*nv-2, dtype=torch.float64)
    Cv1d=torch.sparse_coo_tensor(
        indices = torch.stack([I1D, J1D]),
        values = val1D,
        size = (nv-1, 2*nv),
        dtype = torch.float64
    )
    
    Cv1d = Cv1d.to_dense()
    Ctheta1d = torch.zeros((2 * nv, 2))
    Ctheta1d[0, 0] = 1
    Ctheta1d[-1, 1] = 1

    # build the 1D A matrix

    Av1d = torch.zeros((2 * nv, 2 * nv))

    for i in range(0, nv):
        str1d = torch.tensor([2 * i, 2 * i + 1])  # 0-based indexing
        Ae1d = (1 / hydraulic1d) * (len_edge_frac / 6) * torch.tensor([[2, -1], [-1, 2]])
        Av1d[torch.meshgrid(str1d, str1d, indexing='ij')] = Ae1d.clone()

#     % build matrix M

#     % Assemble
    L_1d = torch.cat([
        torch.cat([Av1d, Bv1d_t, Cv1d.transpose(0, 1), Ctheta1d], dim=1),
        torch.cat([Bv1d_t.T, torch.zeros((ny, ny)), torch.zeros((ny, ny - 1 + 2))], dim=1),
        torch.cat([Cv1d, torch.zeros((ny - 1, ny + ny - 1 + 2))], dim=1),
        torch.cat([torch.zeros((2, 4 * ny - 1)), torch.eye(2)], dim=1)
    ], dim=0)
    
    # Coupled matrices

    Dv_t = [None] * 2
    Dv = [None] * 2

    for k in range(2):
        row_indicesD = torch.arange(0, nv, dtype=torch.int64)  # Equivalent to (1:nv)' in MATLAB
        col_indicesD = RectMesh[k].bd2dofu[subdomain[k].ventcell_bd]  # Already zero-based from bd2dofu
        valuesD = torch.ones(nv, dtype=torch.float64)  # Values in the sparse matrix (all ones)

        # Stack row and column indices for the sparse tensor
        indicesD = torch.stack([row_indicesD, col_indicesD])
        Dv_t[k] = torch.sparse_coo_tensor(
            indicesD,
            values = torch.ones(nv),
            size=(nv, RectMesh[k].NumDofus),
            dtype=torch.float64
        )
        Dv_t[k] = Dv_t[k].to_dense()
        Dv[k] = Dv_t[k].transpose(0, 1)
    
    n_ori_out = [None]*2
    for k in range(2):
        n_ori_out[k] = RectMesh[k].n_ori_out
        
    D_subdom = [None] * 2
    for k in range(2):
        D_subdom[k] = torch.cat([
            Dv[k],
            torch.zeros((n_ori_out[k] - RectMesh[k].NumDofus, nv))
        ])

    D_subdom_t = [D.transpose(0, 1) for D in D_subdom]

    # Assemble 2D and 1D matrix
    L_subdom = [None] * 2
    
    L_subdom[0] = torch.cat([
        RectMesh[0].Stiffmat_Darcy,
        torch.zeros((n_ori_out[0], n_ori_out[1])),
        torch.zeros((n_ori_out[0], 2 * ny)),
        D_subdom[0],
        torch.zeros((n_ori_out[0], ny + 1))
    ], dim=1)
    
    L_subdom[1] = torch.cat([
        torch.zeros((n_ori_out[1], n_ori_out[0])),
        RectMesh[1].Stiffmat_Darcy,
        torch.zeros((n_ori_out[1], 2 * ny)),
        D_subdom[1],
        torch.zeros((n_ori_out[1], ny + 1))
    ], dim=1)
    
    D_all = torch.cat([
        torch.cat([torch.zeros((2 * nv, n_ori_out[0])), torch.zeros((2 * nv, n_ori_out[1]))], dim=1),
        torch.cat(D_subdom_t, dim=1),
        torch.cat([torch.zeros((nv + 1, n_ori_out[0])), torch.zeros((nv + 1, n_ori_out[1]))], dim=1)
    ], dim=0)
    L_all = torch.cat([
        torch.cat(L_subdom, dim=0),
        torch.cat([D_all, L_1d], dim=1)
    ], dim=0)
    
    L_dense = L_all.to_dense() if L_all.is_sparse else L_all
    return L_dense


def DarcySolver(RectMesh, L, nSub, ny):
    n_ori_out = [None]*2
    for k in range(2):
        n_ori_out[k] = RectMesh[k].n_ori_out
    # Initial condition and RHS
    rhs_Z = [None] * 3
    G = [None] * 3
    Gtheta = [None] * 3
    local_all_bc = [None] * 2

    # Initialize G for each subdomain
    for k in range(nSub):
        G[k] = torch.zeros(RectMesh[k].NumDofus, dtype=torch.float64)

    G[2] = torch.zeros(2 * ny)

    # Initialize local boundary conditions
    local_all_bc[0] = torch.cat([
        RectMesh[0].bdLbc,
        RectMesh[0].bdBbc,
        RectMesh[0].bdTbc
    ], dim=0)

    local_all_bc[1] = torch.cat([
        RectMesh[1].bdRbc,
        RectMesh[1].bdBbc,
        RectMesh[1].bdTbc
    ], dim=0)
    
    # Initialize Gtheta for each subdomain
    for k in range(nSub):
        Gtheta[k] = local_all_bc[k][:, 1]
    
    
    Gtheta[2] = torch.tensor([3.0, 0.0])  # Example values, replace if needed

        # Initialize rhs_F for each subdomain
    rhs_F = [None] * 3
    for k in range(2):
        rhs_F[k] = torch.zeros(RectMesh[k].NumEms, dtype=torch.float64)

    rhs_F[2] = torch.zeros(ny, dtype=torch.float64)

    # Assemble rhs_Z for each subdomain
    for k in range(nSub):
        rhs_Z[k] = torch.cat([
            G[k],
            rhs_F[k],
            torch.zeros(RectMesh[k].NumIntEgs, dtype=torch.float64),
            Gtheta[k]
        ], dim=0)

    rhs_Z[2] = torch.cat([
        G[2],
        rhs_F[2],
        torch.zeros(ny - 1, dtype=torch.float64),
        Gtheta[2]
    ], dim=0)

    # Assemble the global RHS vector
    Z = torch.cat(rhs_Z, dim=0)

    L_numpy = L.numpy()  # Convert PyTorch tensor to NumPy array
    L_sparse = csc_matrix(L_numpy)  # Convert NumPy dense matrix to SciPy sparse format

    # Perform LU decomposition using SciPy
    lu = splu(L_sparse)  # LU decomposition of the sparse matrix

    # Convert PyTorch vector Z to NumPy
    Z_numpy = Z.numpy()

    # Solve the linear system Lx = Z using the LU decomposition
    Wf_numpy = lu.solve(Z_numpy)  # SciPy solves the system

    # Convert the solution back to PyTorch
    Wf = torch.from_numpy(Wf_numpy).double()

    # Decompose the solution
    Wf_decomp = [None] * 3
    Wf_decomp[0] = Wf[:n_ori_out[0]]
    Wf_decomp[1] = Wf[n_ori_out[0]:n_ori_out[0] + n_ori_out[1]]
    Wf_decomp[2] = Wf[n_ori_out[0] + n_ori_out[1]:]
    
    # Extract Darcy velocities for each subdomain
    Darcy_u = [None] * 3
    Darcy_u[0] = Wf_decomp[0][:RectMesh[0].NumDofus]
    Darcy_u[1] = Wf_decomp[1][:RectMesh[1].NumDofus]
    Darcy_u[2] = Wf_decomp[2][:2 * ny]
    
    return Wf_decomp, Darcy_u

def localA(invKele,hx,hy):
    # For diffusion D=dI
    a1=invKele;
    a2=invKele;
    a12=0;

    Ae= torch.zeros(4,4, dtype=torch.float64);
    hxy=hx/hy;
    hyx=hy/hx;
    Ae[0, :] = torch.tensor([a1 / 3 * hxy, -a1 / 6 * hxy, a12 / 4, -a12 / 4], dtype=torch.float64)
    Ae[1, :] = torch.tensor([-a1 / 6 * hxy, a1 / 3 * hxy, -a12 / 4, a12 / 4], dtype=torch.float64)
    Ae[2, :] = torch.tensor([a12 / 4, -a12 / 4, a2 / 3 * hyx, -a2 / 6 * hyx], dtype=torch.float64)
    Ae[3, :] = torch.tensor([-a12 / 4, a12 / 4, -a2 / 6 * hyx, a2 / 3 * hyx], dtype=torch.float64)
    
    return Ae

def rhs_data(func, RectMesh):
    # Initialize F as a zero tensor
    F = torch.zeros(RectMesh["NumEms"], dtype=torch.float64)

    # Quadrature point
    pq = 1 / torch.sqrt(torch.tensor(3.0, dtype=torch.float64))

    # Loop over elements
    for i in range(RectMesh["NumEms"]):
        hx = RectMesh["elem_hx"][i]
        hy = RectMesh["elem_hy"][i]
        xmid = RectMesh["elem_center"][i, 0]
        ymid = RectMesh["elem_center"][i, 1]
        xq = hx / 2 * pq
        yq = hy / 2 * pq

        # Compute F for the current element
        F[i] = (
            hx* hy* (
                func(xq + xmid, yq + ymid)
                + func(xq + xmid, -yq + ymid)
                + func(-xq + xmid, yq + ymid)
                + func(-xq + xmid, -yq + ymid)
            )
            / 4
        )

    return F

def Rfrt_TransportSetting(subdomain, nx, ny, NPX, NPY, nt, t0, T, dt, xa, xb, yc, yd, len_edge_frac, hydraulic1d,\
                         diff_gamma, device):
    n_ori_out = [None] * 2
       
    n_ori = [None]*2 # Replace with actual n_ori values
#     len_edge_frac = (yd - yc) / ny
    dy = len_edge_frac
    
    # Diffusion and hydraulic constant
    nSub = NPX * NPY
    nt_all = [nt, nt]
    
    # Initial condition grids
    xgrid = [None] * 2
    ygrid = [None] * nSub

    xgrid[0] = torch.linspace(xa, (xa + xb) / 2, nx + 1)
    xgrid[1] = torch.linspace((xa + xb) / 2, xb, nx + 1)
    for k in range(nSub):
        ygrid[k] = torch.linspace(yc, yd, ny + 1)

    # Compute midpoints for each grid
    xmid = [None] * 2
    ymid = [None] * nSub
    for k in range(nSub):
        xmid[k] = (xgrid[k][:-1] + xgrid[k][1:]) / 2
        ymid[k] = (ygrid[k][:-1] + ygrid[k][1:]) / 2

    Xmid = [None] * 2
    Ymid = [None] * 2
    for k in range(2):
        Xmid[k], Ymid[k] = torch.meshgrid(xmid[k], ymid[k], indexing='ij')

    # Global mesh
    gnx = nx * NPX
    gny = ny * NPY

    xgrid_all = torch.linspace(xa, xb, gnx + 1)
    ygrid_all = torch.linspace(yc, yd, gny + 1)

    xmid_all = (xgrid_all[:-1] + xgrid_all[1:]) / 2
    ymid_all = (ygrid_all[:-1] + ygrid_all[1:]) / 2

    Xmid_all, Ymid_all = torch.meshgrid(xmid_all, ymid_all, indexing='ij')
    
    gRectMesh_transport = RectMesh()
    gRectMesh_transport.RectMesh_Gen_AdvDiff(xa, xb, gnx, yc, yd, gny, device)
    
    gRectMesh_transport.bdLbc = torch.zeros((gny, 2), dtype = torch.float64)
    gRectMesh_transport.bdRbc = torch.zeros((gny, 2), dtype = torch.float64)
    gRectMesh_transport.bdBbc = torch.zeros((gnx, 2), dtype = torch.float64)
    gRectMesh_transport.bdTbc = torch.zeros((gnx, 2), dtype = torch.float64)

    gRectMesh_transport.bdLbc[:, 0] = 2  # Neumann
    gRectMesh_transport.bdRbc[:, 0] = 2  # Neumann
    gRectMesh_transport.bdBbc[:, 0] = 1  # Dirichlet
    gRectMesh_transport.bdTbc[:, 0] = 1  # Dirichlet
    
    gRectMesh_transport.bdBbc[:, 1] = 3

    all_bc = torch.vstack([
        gRectMesh_transport.bdLbc,
        gRectMesh_transport.bdRbc,
        gRectMesh_transport.bdBbc,
        gRectMesh_transport.bdTbc
    ])
    
    gRectMesh_transport.AuBd = torch.zeros(gRectMesh_transport.NumBdEgs, dtype=torch.float64)

    neumann_bd = [i for i, bc in enumerate(all_bc) if bc[0] == 2]
    
    for k in range(nSub):
        subdomain[k].all_bc = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 2), dtype=torch.float64)
              
        subdomain[k].all_bc[subdomain[k].bd_local, 0] = all_bc[subdomain[k].bd_global, 0]
        subdomain[k].all_bc[subdomain[k].if_local_bd, 0] = 4  # Coupled with interface

        subdomain[k].neumann_bd_transport = [i for i, bc in enumerate(subdomain[k].all_bc) if bc[0] == 2]

        subdomain[k].RectMesh.AuBd = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 4), dtype=torch.float64)
        subdomain[k].RectMesh.Bdtheta2dofu = torch.zeros((subdomain[k].RectMesh.NumBdEgs, 4), dtype=torch.float64)
        subdomain[k].ventcell_bd_transport = []  # Initialize as an empty list
        for i in range(len(subdomain[k].all_bc)):
            if subdomain[k].all_bc[i, 0] == 4:  # Check for Ventcell BCs
                subdomain[k].ventcell_bd_transport.append(i)
                
    subdomain[0].RectMesh.bdLbc_transport = gRectMesh_transport.bdLbc
    subdomain[1].RectMesh.bdRbc_transport = gRectMesh_transport.bdRbc

    subdomain[0].RectMesh.bdTbc_transport[:, 0] = 1
    subdomain[0].RectMesh.bdTbc_transport[:, 1] = gRectMesh_transport.bdTbc[:gnx // 2, 1]

    subdomain[1].RectMesh.bdTbc_transport[:, 0] = 1
    subdomain[1].RectMesh.bdTbc_transport[:, 1] = gRectMesh_transport.bdTbc[gnx // 2:, 1]

    subdomain[0].RectMesh.bdBbc_transport[:, 0] = 1
    subdomain[0].RectMesh.bdBbc_transport[:, 1] = gRectMesh_transport.bdBbc[:gnx // 2, 1]

    subdomain[1].RectMesh.bdBbc_transport[:, 0] = 1
    subdomain[1].RectMesh.bdBbc_transport[:, 1] = gRectMesh_transport.bdBbc[gnx // 2:, 1]  
    
    return subdomain

def LocalStiffMat_AdvDiff(RectMesh, neumann_bd_transport, dt, invPara, poros, nx, ny, k, Darcy_u):
    # Build block stiffness matrix
    # Matrix B
    Ivec = torch.arange(0, RectMesh.NumEms)
    Imat = Ivec.repeat(4, 1).T
    I = Imat.reshape(-1)
    J = torch.arange(0, RectMesh.NumDofus)
    val = -torch.ones(RectMesh.NumDofus)
#     print(I)
    B = torch.sparse_coo_tensor(
        indices=torch.vstack([I, J]),  # Stacking row and column indices
        values=val,
        size=(RectMesh.NumEms, RectMesh.NumDofus),
        dtype=torch.float64
    )
    
    B = B.to_dense()
    # Matrix C
    Icvec = torch.arange(0, RectMesh.NumIntEgs)
    Icmat = Icvec.repeat(2, 1).T
    Ic = Icmat.reshape(-1)
#     print(Ic)
    Jcmat = RectMesh.IntEdge2dofu[:, 1:3]
    Jc = Jcmat.reshape(-1)
    valc = torch.ones(2 * RectMesh.NumIntEgs)
    
    C = torch.sparse_coo_tensor(
        indices=torch.vstack([Ic, Jc]),
        values=valc,
        size=(RectMesh.NumIntEgs, RectMesh.NumDofus),
        dtype=torch.float64
    )
    C = C.to_dense()
    # Create the sparse tensor Ctheta
    ### Matrix Ctheta
    v = torch.zeros(ny, 1, dtype=torch.int64);
    if k == 0:
        for i in range(1, ny+1):  # Loop from 1 to ny (MATLAB-style indexing)
            v[i-1] = i * nx  # MATLAB uses 1-based indexing; Python is 0-based
        u = RectMesh.elem2dofu[v-1, 1]  # Assuming elem2dofu is implemented for PyTorch
    elif k == 1:
        for i in range(1, ny+1):  # Loop from 1 to ny
            v[i-1] = (i - 1) * nx +1  # Corrected indexing for Python
        u = RectMesh.elem2dofu[v-1, 0]

    # Initialize r and remove elements in u
    r = RectMesh.bd2dofu.clone()  # Clone to avoid modifying the original tensor
    r = r[~torch.isin(r, u)]

    RectMesh.bd2dofu_nofrac = r

    # Create sparse matrix Ctheta
    Ithe = RectMesh.bd2dofu_nofrac  # Indices of rows
    Jthe = torch.arange(0, RectMesh.NumBdEgs - ny, dtype=torch.float64)  # Indices of columns
    valthe = torch.ones(RectMesh.NumBdEgs - ny, dtype=torch.float64)  # Values

    # Construct sparse tensor Ctheta
    indices = torch.stack([Ithe, Jthe])  # Combine row and column indices
    Ctheta = torch.sparse_coo_tensor(indices, valthe, 
                                     (RectMesh.NumDofus, RectMesh.NumBdEgs - ny),
                                     dtype=torch.float64)
#     print(Ctheta)
    Ctheta = Ctheta.to_dense()
    # Initialize sparse matrices and vectors
    A = torch.zeros((RectMesh.NumDofus, RectMesh.NumDofus), dtype=torch.float64)
    Au = torch.zeros((RectMesh.NumDofus, RectMesh.NumIntEgs), dtype=torch.float64)
    Au_c = torch.zeros(RectMesh.NumDofus, dtype=torch.float64)
    Atheta_nofrac = torch.zeros((RectMesh.NumDofus, RectMesh.NumBdEgs-ny), dtype=torch.float64)
    Atheta_frac = torch.zeros((RectMesh.NumDofus, ny), dtype=torch.float64)
    
    bound_frac_edge_1 = torch.arange(ny + 1, 2 * ny + 1)
    bound_frac_edge_2 = torch.arange(1, ny + 1)
    
    for i in range(RectMesh.NumEms):
        str_ = RectMesh.elem2dofu[i, :]
        
        # Compute local stiffness matrix
        Ae = localA(invPara[i], RectMesh.elem_hx[i], RectMesh.elem_hy[i])
       
        A[torch.meshgrid(str_, str_, indexing='ij')] = Ae.clone()
        veloc_rate_c = torch.tensor([Darcy_u[s] * (1 if Darcy_u[s] >= 0 else -1) for s in str_])
        veloc_rate_theta = torch.tensor([2 * Darcy_u[s] if Darcy_u[s] < 0 else 0 for s in str_])
        
        # Update Au_c with velocity rates
        Au_c[str_] = veloc_rate_c
        
        strLag = RectMesh.dofu2edge[str_]
        veloc_rate_m = veloc_rate_theta.repeat(4).reshape(4, 4)  # Repeat to match Ae dimensions
        Aeu = Ae * veloc_rate_m
        # Boundary and internal indices
        bdindex_all = RectMesh.edge2boundaryforMat[strLag]
        inter_id = torch.where(bdindex_all == 0)[0]  # Internal edges

        # Handle different cases for boundary fractures
        if k == 0:
            idx_1 = ~torch.isin(bdindex_all, bound_frac_edge_1)
            bd_nofrac_id = torch.where((bdindex_all != 0) & idx_1)[0]

            idx_frac_1 = torch.isin(bdindex_all, bound_frac_edge_1)
            bd_frac_id = torch.where((bdindex_all != 0) & idx_frac_1)[0]
        elif k == 1:
            idx_2 = ~torch.isin(bdindex_all, bound_frac_edge_2)
            bd_nofrac_id = torch.where((bdindex_all != 0) & idx_2)[0]

            idx_frac_2 = torch.isin(bdindex_all, bound_frac_edge_2)
            bd_frac_id = torch.where((bdindex_all != 0) & idx_frac_2)[0]
        
        # Map internal edges to indices
        inter_indj = RectMesh.Edge2IntEdge[strLag[inter_id]]
        Au[torch.meshgrid(str_, inter_indj, indexing='ij')] = Aeu[:, inter_id].clone()

        # Boundary indices excluding fractures
        bd_nofrac_index = RectMesh.edge2boundary[strLag[bd_nofrac_id]]

        # Fracture indices
        bd_frac_index = RectMesh.edge2boundary[strLag[bd_frac_id]]
        
        # Adjust boundary indices
        bd_nofrac_index = torch.where(bd_nofrac_index > ny-1, bd_nofrac_index - ny, bd_nofrac_index)
        bd_frac_index = torch.where(bd_frac_index > ny-1, bd_frac_index - ny, bd_frac_index)

        # Update Atheta matrices
        Atheta_nofrac[torch.meshgrid(str_, bd_nofrac_index, indexing='ij')] = Aeu[:, bd_nofrac_id].clone()
        Atheta_frac[torch.meshgrid(str_, bd_frac_index, indexing='ij')] = Aeu[:, bd_frac_id].clone()
    
    Atheta_nofrac *= (0.5)
    Atheta_frac *= (0.5)
    
    # Boundary updates for `Au_c`  
    for i in range(ny):
        str_ = RectMesh.elem2dofu[(i+1)* nx-1, 1]  # 1-based to 0-based
        Darcy_val = Darcy_u[str_]
        Au_c[str_] = Darcy_val if Darcy_val >= 0 else 0

    for i in range(nx):
        str_ = RectMesh.elem2dofu[i, 2]   # 1-based to 0-based
        Darcy_val = Darcy_u[str_]
        Au_c[str_] = Darcy_val if Darcy_val >= 0 else 0

    for i in range(ny):
        str_ = RectMesh.elem2dofu[i*nx, 0]   # 1-based to 0-based
        Darcy_val = Darcy_u[str_]
        Au_c[str_] = Darcy_val if Darcy_val >= 0 else 0

    for i in range(nx):
        str_ = RectMesh.elem2dofu[(ny - 1) * nx + i, 3]   # 1-based to 0-based
        Darcy_val = Darcy_u[str_]
        Au_c[str_] = Darcy_val if Darcy_val >= 0 else 0
    
    Au_c_mat = torch.sparse_coo_tensor(
        indices=torch.vstack([I, J]),  # Stack row and column indices
        values= Au_c,  # Values of the matrix
        size=(RectMesh.NumEms, RectMesh.NumDofus),  # Shape of the matrix
        dtype=torch.float64
    )

    Au_c_mat = Au_c_mat.to_dense()
    
    S=RectMesh.elem_hx*RectMesh.elem_hy   
    n_ori=RectMesh.NumDofus+RectMesh.NumEms+RectMesh.NumIntEgs
    
    M = poros * torch.diag_embed(S)

    RectMesh.M_transport = M

    n_ori = RectMesh.NumDofus + RectMesh.NumEms + RectMesh.NumIntEgs
    zero1 = torch.zeros((RectMesh.NumEms, RectMesh.NumIntEgs + RectMesh.NumBdEgs-ny))
    zero2 = torch.zeros((RectMesh.NumIntEgs, RectMesh.NumEms + RectMesh.NumIntEgs+RectMesh.NumBdEgs-ny))
    zero3 = torch.zeros((RectMesh.NumBdEgs-ny, n_ori))
    
    L = torch.cat([
        torch.cat([A, (B.T - Au_c_mat.T), (C.T - Au), (Ctheta - Atheta_nofrac)], dim=1),
        torch.cat([dt * B, -M, zero1], dim=1),
        torch.cat([C, zero2], dim=1),
        torch.cat([zero3, torch.eye(RectMesh.NumBdEgs-ny)], dim=1)
    ], dim=0)

#     RectMesh.L = L;
    RectMesh.n_ori = n_ori;

    nall = L.shape[0]

    # Apply boundary conditions for Neumann boundaries
    nneu = len(neumann_bd_transport)
    for i in range(nneu):
        current_bd = neumann_bd_transport[i]
        bddofu = RectMesh.bd2dofu[current_bd]
        if current_bd <= ny-1:
            posi = n_ori + current_bd
        else:
            posi = n_ori + current_bd - ny

        L[posi, :] = torch.zeros(nall)  # Set the entire row to zero
        L[posi, bddofu] = 1 
    
    L_dense = L.to_dense() if L.is_sparse else L
 
    return L_dense, Atheta_frac

# def Rfrt_Transport_StiffMat(subdomain, nSub, nx, ny, dy, Darcy_u, dt, Dele, len_edge_frac, delta, diff_gamma, poros_frac):
def Rfrt_Transport_StiffMat(subdomain, nSub, nx, ny, dy, Darcy_u, dt, Dele, len_edge_frac, delta, alp_frt, poros_frac):
    nv = ny
    Atheta_frac = [None]*nSub
    L_transport = [None]*nSub
    n_ori_out = [None] *nSub
    RectMesh = [None]*nSub
    for k in range(nSub):
        RectMesh[k] = subdomain[k].RectMesh
        n_ori_out[k] = RectMesh[k].n_ori_out
    hy = RectMesh[1].elem_hy
    
    invDele = [None]*nSub
#     invDele_m = [None]*nSub
    
    for kk in range(nSub):
        invDele[kk] = torch.zeros(nx*ny, dtype = torch.float64)
    
    for kkk in range(nSub):
        invDele[kkk][:] += 1/ Dele[kkk]
    
    
    
    n_ori = [None]*nSub
    for k in range(nSub):
        L_transport[k], Atheta_frac[k] = \
                    LocalStiffMat_AdvDiff(subdomain[k].RectMesh, subdomain[k].neumann_bd_transport,\
                                          dt, invDele[k], subdomain[k].poros, nx, ny, k, Darcy_u[k])
    
    # Build the 1D B matrix
    Ivent = torch.arange(nv).repeat(2, 1).T.flatten()
    Jvent = torch.arange(2 * nv)
    Bv1d = torch.sparse_coo_tensor(
        indices=torch.vstack([Ivent, Jvent]),
        values=-torch.ones(2 * nv),
        size=(nv, 2 * nv)
    )
    Bv1d = Bv1d.to_dense()
    Bv1d_t = Bv1d.transpose(0, 1)

    # Build the 1D C matrix
    Ivent_c = torch.arange(nv - 1).repeat(2, 1).T.flatten()
    Jvent_c = torch.arange(1, 2 * nv - 1)
    Cv1d = torch.sparse_coo_tensor(
        indices=torch.vstack([Ivent_c, Jvent_c]),
        values=torch.ones(2 * (nv - 1)),
        size=(nv - 1, 2 * nv)
    )
    Cv1d = Cv1d.to_dense()
    
    Ctheta1d = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 2 * nv - 1], [0, 1]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2 * nv, 2)
    )    
    Ctheta1d = Ctheta1d.to_dense()
    # Build the 1D A matrix
    Av1d = torch.zeros(2 * nv, 2 * nv, dtype=torch.float64)
    Au1d = torch.zeros(2 * nv, nv - 1, dtype=torch.float64)
    Atheta1d = torch.zeros(2 * nv, 2, dtype=torch.float64)
    Au_c_1d = torch.zeros(2 * nv, dtype=torch.float64)
    
    edge2boundary1d = torch.cat([torch.tensor([1]), torch.zeros(nv-1), torch.tensor([2])])
    Edge2IntEdge1d = torch.cat([torch.tensor([0]), torch.arange(0, nv-1), torch.tensor([0])])
    
    edge2boundary1d = edge2boundary1d.to(torch.int64)
    Edge2IntEdge1d = Edge2IntEdge1d.to(torch.int64)
    for i in range(nv):
        str1d = torch.tensor([2 * i, 2 * i + 1], dtype=torch.int64)
        Ae1d = (1 / alp_frt) * (len_edge_frac / 6) * torch.tensor([[2, -1], [-1, 2]], dtype=torch.float64)
        Av1d[torch.meshgrid(str1d, str1d, indexing = 'ij')] = Ae1d.clone()

        # Compute velocity rates
        velocity = Darcy_u[2][str1d]
        veloc1d_rate_theta =  2 * velocity * (velocity < 0)
        veloc1d_rate_theta = veloc1d_rate_theta.to(dtype=torch.float64)
        
        veloc1d_rate_c = velocity * ((velocity >= 0).float() - (velocity < 0).float())
        
        Au_c_1d[str1d] = veloc1d_rate_c

        # Update Au1d and Atheta1d
        strLag1d = torch.tensor([i, i + 1], dtype=torch.int64)
        veloc1d_rate_m = veloc1d_rate_theta.repeat(2, 1)
        Aeu1d = Ae1d * veloc1d_rate_m

        bdindex1d_all = edge2boundary1d[strLag1d]
        inter1d_id = (bdindex1d_all == 0).nonzero(as_tuple=True)[0]
        bd1d_id = (bdindex1d_all != 0).nonzero(as_tuple=True)[0]

        inter1d_indj = Edge2IntEdge1d[strLag1d[inter1d_id]]
        Au1d[torch.meshgrid(str1d, inter1d_indj, indexing = 'ij')] = Aeu1d[:, inter1d_id].clone()
        
        bdindex1d = edge2boundary1d[strLag1d[bd1d_id]]

        Atheta1d[torch.meshgrid(str1d, bdindex1d-1, indexing = 'ij')] = Aeu1d[:, bd1d_id].clone()
    
#     print(Au1d)
    Au_c_mat1d = torch.sparse_coo_tensor(
        indices=torch.vstack([Ivent, Jvent]),
        values=Au_c_1d,
        size=(nv, 2 * nv),
        dtype=torch.float64
    )
    Au_c_mat1d = Au_c_mat1d.to_dense()
    # Build matrix M
    diagonal1d = dy*torch.ones(nv, dtype = torch.float64)
    Mv1d = poros_frac * torch.diag(diagonal1d)

    # Assemble matrix L_1d
    L_1d = torch.cat([
        torch.cat([Av1d, Bv1d_t - Au_c_mat1d.T, Cv1d.T - Au1d, Ctheta1d - Atheta1d], dim=1),
        torch.cat([dt * Bv1d, -Mv1d, torch.zeros((nv, nv - 1 + 2))], dim=1),
        torch.cat([Cv1d, torch.zeros((nv - 1, nv + nv - 1 + 2))], dim=1),
        torch.cat([torch.zeros((2, 4 * nv - 1)), torch.eye(2)], dim=1)
    ], dim=0)
    
    Dv_t = [None] * 2
    Dv = [None] * 2

    for k in range(2):
        row_indicesD = torch.arange(0, nv, dtype=torch.int64)  # Equivalent to (1:nv)' in MATLAB
        col_indicesD = RectMesh[k].bd2dofu[subdomain[k].ventcell_bd_transport]  # Already zero-based from bd2dofu
        valuesD = torch.ones(nv, dtype=torch.float64)  # Values in the sparse matrix (all ones)

        # Stack row and column indices for the sparse tensor
        indicesD = torch.stack([row_indicesD, col_indicesD])
        Dv_t[k] = torch.sparse_coo_tensor(
            indicesD,
            values = torch.ones(nv),
            size=(nv, RectMesh[k].NumDofus),
            dtype=torch.float64
        )
        Dv_t[k] = Dv_t[k].to_dense()
        Dv[k] = Dv_t[k].transpose(0, 1)
    
    n_ori_out = [None]*2
    for k in range(2):
        n_ori_out[k] = RectMesh[k].n_ori_out
        
    D_subdom = [None] * 2
    for k in range(2):
        D_subdom[k] = torch.cat([
            Dv[k]-Atheta_frac[k],
            torch.zeros((n_ori_out[k] - RectMesh[k].NumDofus, nv))
        ])

    D_subdom_t = [D.transpose(0, 1) for D in D_subdom]

    # Assemble 2D and 1D matrix
    L_subdom = [None] * 2
    
    L_subdom[0] = torch.cat([
        L_transport[0],
        torch.zeros((n_ori_out[0], n_ori_out[1])),
        torch.zeros((n_ori_out[0], 2 * ny)),
        D_subdom[0],
        torch.zeros((n_ori_out[0], ny + 1))
    ], dim=1)
    
    L_subdom[1] = torch.cat([
        torch.zeros((n_ori_out[1], n_ori_out[0])),
        L_transport[1],
        torch.zeros((n_ori_out[1], 2 * ny)),
        D_subdom[1],
        torch.zeros((n_ori_out[1], ny + 1))
    ], dim=1)
    
    D_all = torch.cat([
        torch.cat([torch.zeros((2 * nv, n_ori_out[0])), torch.zeros((2 * nv, n_ori_out[1]))], dim=1),
        torch.cat([dt*D_subdom_t[0], dt*D_subdom_t[1]], dim=1),
        torch.cat([torch.zeros((nv + 1, n_ori_out[0])), torch.zeros((nv + 1, n_ori_out[1]))], dim=1)
    ], dim=0)
    
    L_subdom[0] = L_subdom[0].to_sparse()
    L_subdom[1] = L_subdom[1].to_sparse()
    
    D_all = D_all.to_sparse()
    L_1d = L_1d.to_sparse()
    
    L_transport = torch.cat([
        torch.cat(L_subdom, dim=0),
        torch.cat([D_all, L_1d], dim=1)
    ], dim=0)
    
    return L_transport, Mv1d

def Rfrt_Transport_Solver(p0, subdomain, Mv1d, L, nSub, nx, ny, Darcy_u, dt, delta, poros_frac):
    RectMesh = [None]*nSub
    M_transport = [None]*nSub
    n_ori_out = [None]*nSub
    
    for k in range(nSub):
        RectMesh[k] = subdomain[k].RectMesh
        M_transport[k] = RectMesh[k].M_transport
        n_ori_out[k] = RectMesh[k].n_ori_out
        
    rhs_F = [None] * 3
    rhs_Z = [None] * 3
    G = [None] * 3
    Gtheta = [None] * 3
    local_all_bc = [None] * 2    
    
    # Initialize G for all subdomains
    G[0] = torch.zeros(RectMesh[0].NumDofus, dtype=torch.float64)
    G[1] = torch.zeros(RectMesh[1].NumDofus, dtype=torch.float64)
    G[2] = torch.zeros(2*ny, dtype=torch.float64)

    # Compute rhs_F for all subdomains
    rhs_F[0] = -torch.matmul(M_transport[0], p0[0])
    rhs_F[1] = -torch.matmul(M_transport[1], p0[1])
    rhs_F[2] = -torch.matmul(Mv1d, p0[2])
   
    # Apply Dirichlet BCs
    local_all_bc[0] = torch.cat([
        RectMesh[0].bdLbc_transport,
        RectMesh[0].bdBbc_transport,
        RectMesh[0].bdTbc_transport
    ], dim=0)

    local_all_bc[1] = torch.cat([
        RectMesh[1].bdRbc_transport,
        RectMesh[1].bdBbc_transport,
        RectMesh[1].bdTbc_transport
    ], dim=0)

    # Assign Gtheta values
    Gtheta[0] = local_all_bc[0][:, 1]  # [:,2] in MATLAB, so [:,1:2] to preserve dimensions
    Gtheta[1] = local_all_bc[1][:, 1]
    Gtheta[2] = torch.tensor([3, 0])

    # Compute rhs_Z for all subdomains
    zeros_size = [None]*3
    zeros_size[0] = RectMesh[0].NumIntEgs
    zeros_size[1] = RectMesh[1].NumIntEgs
    zeros_size[2] = ny-1
    
    for k in range(3):
        if rhs_F[k].dim() == 1: 
            rhs_Z[k] = torch.cat([G[k], rhs_F[k], torch.zeros(zeros_size[k], dtype=torch.float64), Gtheta[k]], dim=0)
        else:
            batch_size = rhs_F[k].shape[1]  # Get the batch size (N)

            # Expand the other tensors to match the batch size
            G_batched = G[k].unsqueeze(1).expand(-1, batch_size)  # Expand G[0] to (len(G[0]), N)
            zeros_batched = torch.zeros(zeros_size[k], batch_size, dtype=torch.float64)  # Create a zero tensor (NumIntEgs, N)
            Gtheta_batched = Gtheta[k].unsqueeze(1).expand(-1, batch_size)  # Expand Gtheta[0] to (len(Gtheta[0]), N)

            # Concatenate along the 0th dimension
            rhs_Z[k] = torch.cat([G_batched, rhs_F[k], zeros_batched, Gtheta_batched], dim=0)
    
    # Combine rhs_Z into Z
    Z = torch.cat([rhs_Z[0], rhs_Z[1], rhs_Z[2]], dim=0)
    
    L_numpy = L.to_dense().numpy()  # Convert PyTorch tensor to NumPy array
    L_sparse = csc_matrix(L_numpy)  # Convert NumPy dense matrix to SciPy sparse format
    # Perform LU decomposition using SciPy


#     # Convert PyTorch vector Z to NumPy
    Z_numpy = Z.numpy()
    Wf_numpy = spsolve(L_sparse, Z_numpy)
#     # Solve the linear system Lx = Z using the LU decomposition
#     # Convert the solution back to PyTorch
    Wf = torch.from_numpy(Wf_numpy).double()

#     Decompose the solution
    Wf_decomp = [None] * 3
    Wf_decomp[0] = Wf[:n_ori_out[0]]
    Wf_decomp[1] = Wf[n_ori_out[0]:n_ori_out[0] + n_ori_out[1]]
    Wf_decomp[2] = Wf[n_ori_out[0] + n_ori_out[1]:]
    
    return Wf

def Rfrt_Transport_Solver_DF(nt, n_dim, subdomain, Mv1d, L, nSub, nx, ny, Darcy_u, dt, delta, poros_frac):
    RectMesh = [None]*nSub
    M_transport = [None]*nSub
    n_ori_out = [None]*nSub
    NumDofus = [None]*nSub
    NumEms = [None]*nSub
    
    for k in range(nSub):
        RectMesh[k] = subdomain[k].RectMesh
        M_transport[k] = RectMesh[k].M_transport
        n_ori_out[k] = RectMesh[k].n_ori_out
        NumDofus[k] = RectMesh[k].NumDofus
        NumEms[k] = RectMesh[k].NumEms

    p0DF = [None]*3
    p0DF[0] = torch.zeros((RectMesh[0].NumEms, ), dtype=torch.float64)
    p0DF[1] = torch.zeros((RectMesh[1].NumEms, ), dtype=torch.float64)
    p0DF[2] = torch.zeros((ny, ), dtype=torch.float64)
    
    rhs_F = [None] * 3
    rhs_Z = [None] * 3
    G = [None] * 3
    Gtheta = [None] * 3
    local_all_bc = [None] * 2    
    
    # Initialize G for all subdomains
    G[0] = torch.zeros(RectMesh[0].NumDofus, dtype=torch.float64)
    G[1] = torch.zeros(RectMesh[1].NumDofus, dtype=torch.float64)
    G[2] = torch.zeros(2*ny, dtype=torch.float64)

    # Apply Dirichlet BCs
    local_all_bc[0] = torch.cat([
        RectMesh[0].bdLbc_transport,
        RectMesh[0].bdBbc_transport,
        RectMesh[0].bdTbc_transport
    ], dim=0)

    local_all_bc[1] = torch.cat([
        RectMesh[1].bdRbc_transport,
        RectMesh[1].bdBbc_transport,
        RectMesh[1].bdTbc_transport
    ], dim=0)

    # Assign Gtheta values
    Gtheta[0] = local_all_bc[0][:, 1]  # [:,2] in MATLAB, so [:,1:2] to preserve dimensions
    Gtheta[1] = local_all_bc[1][:, 1]
    Gtheta[2] = torch.tensor([3, 0])

    zeros_size = [None]*3
    zeros_size[0] = RectMesh[0].NumIntEgs
    zeros_size[1] = RectMesh[1].NumIntEgs
    zeros_size[2] = ny-1

    L_numpy = L.to_dense().numpy()  # Convert PyTorch tensor to NumPy array
    L_sparse = csc_matrix(L_numpy)  # Convert NumPy dense matrix to SciPy sparse format

    Wf = torch.zeros((n_dim, nt+1), dtype = torch.float64)
    
    for i in range(nt):
        rhs_F[0] = -torch.matmul(M_transport[0], p0DF[0])
        rhs_F[1] = -torch.matmul(M_transport[1], p0DF[1])
        rhs_F[2] = -torch.matmul(Mv1d, p0DF[2])
        
        for k in range(3):
            if rhs_F[k].dim() == 1: 
                rhs_Z[k] = torch.cat([G[k], rhs_F[k], torch.zeros(zeros_size[k], dtype=torch.float64), Gtheta[k]], dim=0)
            else:
                batch_size = rhs_F[k].shape[1]  # Get the batch size (N)
    
                # Expand the other tensors to match the batch size
                G_batched = G[k].unsqueeze(1).expand(-1, batch_size)  # Expand G[0] to (len(G[0]), N)
                zeros_batched = torch.zeros(zeros_size[k], batch_size, dtype=torch.float64)  # Create a zero tensor (NumIntEgs, N)
                Gtheta_batched = Gtheta[k].unsqueeze(1).expand(-1, batch_size)  # Expand Gtheta[0] to (len(Gtheta[0]), N)
    
                # Concatenate along the 0th dimension
                rhs_Z[k] = torch.cat([G_batched, rhs_F[k], zeros_batched, Gtheta_batched], dim=0)
        
        Z = torch.cat([rhs_Z[0], rhs_Z[1], rhs_Z[2]], dim=0)
    
        Z_numpy = Z.numpy()
        Wf_numpy = spsolve(L_sparse, Z_numpy)

        Wf_numpy = torch.from_numpy(Wf_numpy).double()
        Wf_numpy = Wf_numpy.to(torch.float64)
        Wf[:, i+1] +=Wf_numpy
        
        ## update p0
        p0DF[0] = Wf[NumDofus[0]+torch.arange(0, NumEms[0]), i+1]
        p0DF[1] = Wf[n_ori_out[0]+NumDofus[1]+torch.arange(0, NumEms[1]), i+1]
        p0DF[2] = Wf[n_ori_out[0]+n_ori_out[1]+2*ny+torch.arange(0, ny), i+1]
    
# #     Decompose the solution
    
    return Wf
    
def RearrangeSol(sln, RectMesh, ny):
    ReSln = torch.zeros_like(sln)
    NumDofus = [0]*2
    NumEms = [0]*2
    n_ori_out = [0]*2
    NumIntEgs = [0]*2
    NumBdEgs = [0]*2
    NumLagr = [0]*2
    # print(RectMesh[1].n_ori_out)
    for k in range(2):
        n_ori_out[k] = RectMesh[k].n_ori_out
        NumEms[k] = RectMesh[k].NumEms
        NumDofus[k] = RectMesh[k].NumDofus
        NumIntEgs[k] = RectMesh[k].NumIntEgs
        NumBdEgs[k] = RectMesh[k].NumBdEgs
        NumLagr[k] = NumIntEgs[k]+NumBdEgs[k]-ny
        
    ReSln[:, :NumDofus[0]] += sln[:, :NumDofus[0]]
                             
    ReSln[:, NumDofus[0]+torch.arange(0, NumDofus[1])] +=sln[:, n_ori_out[0]+torch.arange(0, NumDofus[1])]
          
    ReSln[:, NumDofus[0]+NumDofus[1]+torch.arange(0, 2*ny)] += sln[:, n_ori_out[0]+n_ori_out[1]+torch.arange(0, 2*ny)]
                             
    ReSln[:, NumDofus[0]+NumDofus[1]+2*ny+torch.arange(0, NumEms[0])] += sln[:, NumDofus[0]+torch.arange(0, NumEms[0])]
    
    ReSln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+torch.arange(0, NumEms[1])] +=\
                             sln[:, n_ori_out[0]+NumDofus[1]+torch.arange(0, NumEms[1])]
    
    ReSln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+torch.arange(0, ny)] += \
                            sln[:, n_ori_out[0]+n_ori_out[1]+2*ny+torch.arange(0, ny)]
    
    ReSln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+ny+torch.arange(0,NumLagr[0])] += \
                            sln[:, NumDofus[0]+NumEms[0]+torch.arange(0,NumLagr[0])]
    
    ReSln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+ny+NumLagr[0]+torch.arange(0, NumLagr[1])] += \
                            sln[:, n_ori_out[0]+NumDofus[1]+NumEms[1]+torch.arange(0,NumLagr[1])]
    
    ReSln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+ny+NumLagr[0]+NumLagr[1]+torch.arange(0, ny+1)] += \
                            sln[:, n_ori_out[0]+n_ori_out[1]+2*ny+ny+torch.arange(0, ny+1)]
    return ReSln

def RearrangeSol_Reverse(sln, RectMesh, ny):
    ReSln = torch.zeros_like(sln)
    NumDofus = [0]*2
    NumEms = [0]*2
    n_ori_out = [0]*2
    NumIntEgs = [0]*2
    NumBdEgs = [0]*2
    NumLagr = [0]*2
    
    for k in range(2):
        n_ori_out[k] = RectMesh[k].n_ori_out
        NumEms[k] = RectMesh[k].NumEms
        NumDofus[k] = RectMesh[k].NumDofus
        NumIntEgs[k] = RectMesh[k].NumIntEgs
        NumBdEgs[k] = RectMesh[k].NumBdEgs
        NumLagr[k] = NumIntEgs[k]+NumBdEgs[k]-ny
        
    ReSln[:, :NumDofus[0]] += sln[:, :NumDofus[0]]
                             
    ReSln[:, n_ori_out[0]+torch.arange(0, NumDofus[1])] +=sln[:, NumDofus[0]+torch.arange(0, NumDofus[1])]
          
    ReSln[:, n_ori_out[0]+n_ori_out[1]+torch.arange(0, 2*ny)] += sln[:, NumDofus[0]+NumDofus[1]+torch.arange(0, 2*ny)]
                             
    ReSln[:, NumDofus[0]+torch.arange(0, NumEms[0])] += sln[:, NumDofus[0]+NumDofus[1]+2*ny+torch.arange(0, NumEms[0])]
    
    ReSln[:, n_ori_out[0]+NumDofus[1]+torch.arange(0, NumEms[1])] +=\
                             sln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+torch.arange(0, NumEms[1])]
    
    ReSln[:, n_ori_out[0]+n_ori_out[1]+2*ny+torch.arange(0, ny)] += \
                            sln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+torch.arange(0, ny)]
    
    ReSln[:, NumDofus[0]+NumEms[0]+torch.arange(0,NumLagr[0])] += \
                            sln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+ny+torch.arange(0,NumLagr[0])]
    
    ReSln[:, n_ori_out[0]+NumDofus[1]+NumEms[1]+torch.arange(0,NumLagr[1])] += \
                            sln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+ny+NumLagr[0]+torch.arange(0, NumLagr[1])]
       
    ReSln[:, n_ori_out[0]+n_ori_out[1]+2*ny+ny+torch.arange(0, ny+1)] += \
                            sln[:, NumDofus[0]+NumDofus[1]+2*ny+NumEms[0]+NumEms[1]+ny+NumLagr[0]+NumLagr[1]+torch.arange(0, ny+1)]
              
    return ReSln


def DecompSol(sln, RectMesh, ny):
    timeshape = sln.shape[0]
    n_ori_out = [None] * 2
    for k in range(2):
        n_ori_out[k] = RectMesh[k].n_ori_out
    
    Sln1 = torch.zeros(timeshape, n_ori_out[0], dtype=torch.float64)
    Sln2 = torch.zeros(timeshape, n_ori_out[1], dtype=torch.float64)
    Slnfrt = torch.zeros(timeshape, ny+2*ny+ny+1, dtype=torch.float64)
    
    Sln1[:, :] += sln[:, :n_ori_out[0]]
    
    Sln2[:, :] += sln[:, n_ori_out[0]:n_ori_out[0]+n_ori_out[1]]
    
    Slnfrt[:, :] += sln[:, n_ori_out[0]+n_ori_out[1]:n_ori_out[0]+n_ori_out[1]+ny+ny+1+2*ny]
              
    return Sln1, Sln2, Slnfrt