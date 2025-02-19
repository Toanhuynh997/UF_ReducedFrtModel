import torch
import torch.sparse as sparse
from scipy.sparse import csr_matrix
import scipy.sparse
import scipy.sparse.linalg
import math
import numpy as np
import scipy 


class TriMesh:
    def __init__(self):
        self.NumEms = None
        self.NumEgs = None
        self.BndryEdge = None
        self.edge = None
        self.node = None
        self.elem = None
        self.BndryDescMat = None
#         self.EqnBC_Glob = None
        self.DOFs = None
        self.PermK = None
        self.BndryCondType = None
        self.FractureEgs = None
        self.DirichletEgs = None
        self.NeumannEgs = None
        self.flag = 0
        self.NumEgsDiag = 0
        self.NumEgsHori = 0
        self.NumEgsVert = 0
        self.elem2edge = None
        self.edge2elem = None
        self.area = None
        self.LenEg = None
        
    def RectDom_TriMesh_GenUnfm(self, xa, xb, nx, yc, yd, ny, device, status):
#         mesh = TriMesh()

        # Primary Mesh Information
        self.NumNds = (nx + 1) * (ny + 1)
        self.NumEms = 2 * nx * ny

        # Generating nodes
        x = torch.linspace(xa, xb, nx + 1)
        y = torch.linspace(yc, yd, ny + 1)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.node = torch.column_stack((X.flatten(), Y.flatten()))  # Shape: (NumNds, 2)
        self.node = self.node.to(device)
        # Generating elements
        self.elem = torch.zeros((self.NumEms, 3), dtype=torch.int64, device=device)
        for i in range(nx):
            for j in range(ny):
                # Left triangle
                k_left = i * 2 * ny + 2 * j
                self.elem[k_left, 0] = i * (ny + 1) + j
                self.elem[k_left, 1] = self.elem[k_left, 0] + (ny + 1)
                self.elem[k_left, 2] = self.elem[k_left, 0] + 1

                # Right triangle
                k_right = i * 2 * ny + 2 * j + 1
                self.elem[k_right, 0] = (i + 1) * (ny + 1) + (j + 1)
                self.elem[k_right, 1] = self.elem[k_right, 0] - (ny + 1)
                self.elem[k_right, 2] = self.elem[k_right, 0] - 1

        if status == 1:
            self.flag = 1
            return 

        # Secondary Mesh Information
        self.NumEgsDiag = nx * ny
        self.NumEgsHori = nx * (ny + 1)
        self.NumEgsVert = (nx + 1) * ny
        self.NumEgs = self.NumEgsVert + self.NumEgsHori + self.NumEgsDiag

        # Initialize edges
        self.edge = torch.zeros((self.NumEgs, 2), dtype=torch.int64, device=device)

        # Vertical edges
        for i in range(nx + 1):
            for j in range(ny):
                k = i * ny + j
                self.edge[k, 0] = i * (ny + 1) + j
                self.edge[k, 1] = i * (ny + 1) + j + 1

        # Horizontal edges
        for i in range(nx):
            for j in range(ny + 1):
                k = self.NumEgsVert + j * nx + i
                self.edge[k, 0] = i * (ny + 1) + j
                self.edge[k, 1] = (i + 1) * (ny + 1) + j

        # Diagonal edges
        for i in range(nx):
            for j in range(ny):
                k = self.NumEgsVert + self.NumEgsHori + i * ny + j
                self.edge[k, 0] = i * (ny + 1) + j + 1
                self.edge[k, 1] = (i + 1) * (ny + 1) + j

        # Element to Edge mapping
        self.elem2edge = torch.zeros((self.NumEms, 3), dtype=torch.int64, device=device)
        for i in range(nx):
            for j in range(ny):
                # Left triangle
                k_left = i * 2 * ny + 2 * j
                diag_edge = self.NumEgsVert + self.NumEgsHori + i * ny + j
                vert_edge = i * ny + j
                hori_edge = self.NumEgsVert + j * nx + i
                edge_indices1 = torch.tensor([diag_edge, vert_edge, hori_edge], dtype=torch.int64, device=device)
                self.elem2edge[k_left, :] = edge_indices1

                # Right triangle
                k_right = i * 2 * ny + 2 * j + 1
                diag_edge = self.NumEgsVert + self.NumEgsHori + i * ny + j
                vert_edge = (i + 1) * ny + j
                hori_edge = self.NumEgsVert + (j+1) * nx + i
                edge_indices2 = torch.tensor([diag_edge, vert_edge, hori_edge], dtype=torch.int64, device=device)
                self.elem2edge[k_right, :] = edge_indices2

        # Edge to Element mapping
        self.edge2elem = torch.zeros((self.NumEgs, 2), dtype=torch.int64, device=device)
        CntEmsEg = torch.zeros(self.NumEgs, dtype=torch.int64, device=device)
        for ie in range(self.NumEms):
            LblEg = self.elem2edge[ie, :3]
            CntEmsEg[LblEg] += 1
            for k in range(3):
                self.edge2elem[LblEg[k], CntEmsEg[LblEg[k]]-1] = ie

        # Calculate areas of all elements
        k1 = self.elem[:, 0]
        k2 = self.elem[:, 1]
        k3 = self.elem[:, 2]
        x1 = self.node[k1, 0]
        y1 = self.node[k1, 1]
        x2 = self.node[k2, 0]
        y2 = self.node[k2, 1]
        x3 = self.node[k3, 0]
        y3 = self.node[k3, 1]
        self.area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

        # Calculate lengths of all edges
        e_k1 = self.edge[:, 0]
        e_k2 = self.edge[:, 1]
        ex1 = self.node[e_k1, 0]
        ey1 = self.node[e_k1, 1]
        ex2 = self.node[e_k2, 0]
        ey2 = self.node[e_k2, 1]
        self.LenEg = torch.sqrt((ex2 - ex1)**2 + (ey2 - ey1)**2)

        self.flag = 2
        
    def TriMesh_Enrich2(self, BndryDescMat, device):
        NumEgs = self.NumEgs
        self.BndryEdge = torch.zeros(NumEgs, dtype=torch.int64, device=device)
        atol = 1e-9
        rtol = 1e-9

        # Loop over all edges
        for ig in range(NumEgs):
            # Skip interior edges (where edge2elem has a non-zero second column)
            if self.edge2elem[ig, 1] > 0:
                continue

            # Extract node indices for the edge (assuming 0-based indexing)
            node1 = self.edge[ig, 0]
            node2 = self.edge[ig, 1]

            # Extract coordinates for the nodes
            x1, y1 = self.node[node1, 0], self.node[node1, 1]
            x2, y2 = self.node[node2, 0], self.node[node2, 1]

            # Screening with the boundary description matrix
            for k in range(BndryDescMat.shape[0]):
                X1, Y1, X2, Y2 = BndryDescMat[k, 0], BndryDescMat[k, 1], BndryDescMat[k, 2], BndryDescMat[k, 3]

                # Calculate cross products to determine alignment
                lhs1 = (x1 - X1) * (Y2 - Y1)
                rhs1 = (X2 - X1) * (y1 - Y1)
                lhs2 = (x2 - X1) * (Y2 - Y1)
                rhs2 = (X2 - X1) * (y2 - Y1)

                # Use torch.isclose for floating-point comparison
                if (torch.isclose(lhs1, rhs1, atol=atol, rtol=rtol)) and \
                   (torch.isclose(lhs2, rhs2, atol=atol, rtol=rtol)):
                    self.BndryEdge[ig] = k + 1  # MATLAB is 1-based, Python is 0-based
                    break  # Exit the inner loop once a match is found

        return self
    
    def TriMesh_Enrich3(self, BndryDescMat, device):
        NumEms = self.NumEms
        NumEgs = self.NumEgs

        # Element centers
        k1, k2, k3 = self.elem[:, 0], self.elem[:, 1], self.elem[:, 2]
        x1, y1 = self.node[k1, 0], self.node[k1, 1]
        x2, y2 = self.node[k2, 0], self.node[k2, 1]
        x3, y3 = self.node[k3, 0], self.node[k3, 1]
        xc = (1.0 / 3) * (x1 + x2 + x3)
        yc = (1.0 / 3) * (y1 + y2 + y3)
        EmCntr = torch.stack((xc, yc), dim=1)

        # Edge normal and tangential vectors
        AllEg = self.node[self.edge[:, 1], :]-self.node[self.edge[:, 0], :]
        LenEg = torch.sqrt(AllEg[:, 0] ** 2 + AllEg[:, 1] ** 2)
        TanEg = AllEg / LenEg.unsqueeze(1)
        NmlEg = torch.stack((TanEg[:, 1], -TanEg[:, 0]), dim=1)  # Rotate 90 degrees clockwise
        
        NmlEg = NmlEg.to(torch.float64)
        TanEg = TanEg.to(torch.float64)
        
        ig = torch.where(self.BndryEdge > 0)[0]
        NmlEg[ig, :] = BndryDescMat[self.BndryEdge[ig] - 1, 4:6]
        TanEg[ig, :] = torch.stack((NmlEg[ig, 1], -NmlEg[ig, 0]), dim=1)

        # Determine which edge belongs to each element
        row_indices = torch.repeat_interleave(torch.arange(NumEms), 3)  # Repeat row indices for each edge
        col_indices = self.elem2edge.flatten()  # Flatten the elem2edge to get column indices
        data = torch.tile(torch.arange(1, 4), (NumEms,))  # Create data array with values 1, 2, 3 for each edge
        row_indices = row_indices.to(device)
        col_indices = col_indices.to(device)
        data = data.to(device)
#         data 
#         if data.is_cuda: 
#             AuxMat = csr_matrix((data.cpu().numpy(), (row_indices.cpu().numpy(), col_indices.cpu().numpy())), shape=(NumEms, NumEgs))
#         else:
#             AuxMat = csr_matrix((data.numpy(), (row_indices.numpy(), col_indices.numpy())), shape=(NumEms, NumEgs))
        AuxMat = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),  # Stack row and col indices
                values=data,
                size=(NumEms, NumEgs),
                device=device  # Keep the tensor on the same device (GPU)
            )
        WhichEdge = torch.zeros((NumEgs, 2), dtype=torch.int64, device=device)
#         if AuxMat.is_cuda:
#             print("The AuxMat is on GPU (CUDA)")
#         else:
#             print("The AuxMat is on CPU")
            
            
        for ig in range(NumEgs):
            WhichEdge[ig, 0] = AuxMat[self.edge2elem[ig, 0], ig]
            if self.edge2elem[ig, 1] > 0:
                WhichEdge[ig, 1] = AuxMat[self.edge2elem[ig, 1], ig]
                
#         WhichEdge = WhichEdge.to(device)
        
        # Edge midpoints
        xm = torch.zeros((NumEms, 3), dtype=torch.float64, device=device)
        ym = torch.zeros((NumEms, 3), dtype=torch.float64, device=device)
        xm[:, 0] = 0.5 * (x2 + x3)
        ym[:, 0] = 0.5 * (y2 + y3)
        xm[:, 1] = 0.5 * (x3 + x1)
        ym[:, 1] = 0.5 * (y3 + y1)
        xm[:, 2] = 0.5 * (x1 + x2)
        ym[:, 2] = 0.5 * (y1 + y2)

        # Signs for element-wise edge normals
        DirVec1 = torch.stack((xm[:, 0], ym[:, 0]), dim=1) - torch.stack((xc, yc), dim=1)
        DirVec2 = torch.stack((xm[:, 1], ym[:, 1]), dim=1) - torch.stack((xc, yc), dim=1)
        DirVec3 = torch.stack((xm[:, 2], ym[:, 2]), dim=1) - torch.stack((xc, yc), dim=1)

        sn = torch.zeros((NumEms, 3), dtype=torch.float64, device=device)
        sn[:, 0] = torch.sign(torch.einsum('ij,ij->i', DirVec1, NmlEg[self.elem2edge[:, 0], :]))
        sn[:, 1] = torch.sign(torch.einsum('ij,ij->i', DirVec2, NmlEg[self.elem2edge[:, 1], :]))
        sn[:, 2] = torch.sign(torch.einsum('ij,ij->i', DirVec3, NmlEg[self.elem2edge[:, 2], :]))

        # Further signs
        row_indices = torch.repeat_interleave(torch.arange(NumEms), 3)  # Repeat each row index 3 times
        col_indices = self.elem2edge.flatten()  # Flatten elem2edge for all columns
        data = sn.flatten()
        
        row_indices = row_indices.to(device)
        col_indices = col_indices.to(device)
        data = data.to(device)
        # Create the sparse matrix in CSR format
#         AuxMat = csr_matrix((data.numpy(), (row_indices.numpy(), col_indices.numpy())), shape=(NumEms, NumEgs))
        AuxMat = torch.sparse_coo_tensor(
                        indices=torch.stack([row_indices, col_indices], dim=0),  # Stack row and col indices
                        values=data,                                             # Tensor values
                        size=(NumEms, NumEgs) , # Shape of the matrix
                        dtype = torch.float64,
                        device=device
                    )
        sns = torch.zeros((NumEgs, 2), dtype=torch.float64, device=device)
        for ig in range(NumEgs):
            sns[ig, 0] = AuxMat[self.edge2elem[ig, 0], ig].clone().detach()
            if self.edge2elem[ig, 1] > 0:
                sns[ig, 1] = AuxMat[self.edge2elem[ig, 1], ig].clone().detach()

        # Update the TriMesh with tertiary info
        self.EmCntr = EmCntr
        # TriMesh.EgMidPt = EgMidPt (commented out as in the MATLAB code)
        self.NmlEg = NmlEg
        self.TanEg = TanEg
        self.WhichEdge = WhichEdge
        self.SignEmEg = sn
        self.SignEgEm = sns.clone().detach()
        self.flag = 3

        return self
    
class EqnBC:
    def __init__(self):
        self.fxnf = self.fxnf_subdom
        self.fxnK = self.fxnK_func

    def fxnK_func(self, pt):
        """
        Diffusion coefficient or permeability as a 2x2 symmetric positive definite matrix.

        Parameters:
            pt (torch.Tensor): Tensor of points (NumPts x 2)

        Returns:
            K (torch.Tensor): Diffusion coefficient matrix (NumPts x 2 x 2)
        """
        NumPts = pt.shape[0]
        K = torch.zeros((NumPts, 2, 2), dtype=pt.dtype, device=pt.device)
        K[:, 0, 0] = 1.0
        K[:, 1, 1] = 1.0
        return K

    def fxnf_subdom(self, pt, t):
        """
        Function f defined over points pt at time t for subdomains.

        Parameters:
            pt (torch.Tensor): Tensor of points (NumPts x 2)
            t (float): Time variable

        Returns:
            f (torch.Tensor): Evaluated function values (NumPts,)
        """
        x = pt[:, 0]
        y = pt[:, 1]

        # Uncomment one of the following lines to define f as needed

        # Example 1:
        # f = (3 * math.pi**2) * torch.exp(math.pi**2 * t) * torch.sin(math.pi * x) * torch.sin(math.pi * y)

        # Example 2:
        # f = torch.exp(t) * x * (x - 1) * y * (y - 1) - 2 * torch.exp(t) * (x * (x - 1) + y * (y - 1))

        # Default (zero function):
        f = torch.zeros_like(x)

        # Example 3:
        # f = (3 * math.pi**2) * torch.exp(math.pi**2 * t) * torch.cos(math.pi * x) * torch.cos(math.pi * y)

        return f
    
def Darcy_SmplnPerm_TriMesh(fxnK, TriMesh, GAUSSQUAD):
    """
    Assemble the permeability tensor for the Darcy equation using PyTorch.

    Parameters:
        fxnK (callable): Function that takes a tensor of points (NumEms x 2) and returns
                         a tensor of diffusion coefficients (NumEms x 2 x 2).
        TriMesh (TriMesh): The mesh object containing mesh information.
        GAUSSQUAD (dict): Dictionary containing quadrature points and weights. Expected to have a key 'TRIG'.

    Returns:
        PermK (torch.Tensor): Assembled permeability tensor (NumEms x 2 x 2).
    """
    NumEms = TriMesh.NumEms

    # Initialize PermK as a zero tensor on the same device and dtype as TriMesh.node
    PermK = torch.zeros((NumEms, 2, 2), dtype=TriMesh.node.dtype, device=TriMesh.node.device)
    
    # Extract quadrature points and weights
    NumQuadPts = GAUSSQUAD['TRIG'].shape[0]
#     NumQuadPts = NumQuadPts.to(device)
    for k in range(NumQuadPts):
        # Extract barycentric coordinates and quadrature weight
        w0, w1, w2, weight = GAUSSQUAD['TRIG'][k]
        
        # Compute quadrature points for all elements
        qp = (w0 * TriMesh.node[TriMesh.elem[:, 0], :] +
              w1 * TriMesh.node[TriMesh.elem[:, 1], :] +
              w2 * TriMesh.node[TriMesh.elem[:, 2], :])
        
        # Ensure qp is a torch.Tensor
        if not isinstance(qp, torch.Tensor):
            qp = torch.tensor(qp, dtype=TriMesh.node.dtype, device=TriMesh.node.device)
        
        # Compute diffusion coefficients at quadrature points
        K = fxnK(qp)  # Expected shape: (NumEms, 2, 2)
        
        # Update PermK with the weighted diffusion coefficients
        PermK += weight * K

    return PermK


def SetGaussQuad(ch1, ch2, ch3, device):
    GAUSSQUAD = {
        'LINE': [],
        'RECT': [],
        'TRIG': []
    }
    
    # Choice 1 for line segments
    if ch1 == 1:
        # 1-point Gaussian quadrature on a line segment (the center)
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.5, 0.5, 1.0]
        ], dtype=torch.float64, device=device)
    elif ch1 == 2:
        # 2-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.788675134594813, 0.211324865405187, 0.5],
            [0.211324865405187, 0.788675134594813, 0.5]
        ], dtype=torch.float64, device=device)
    elif ch1 == 3:
        # 3-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.88729833462074, 0.11270166537926, 0.27777777777778],
            [0.5, 0.5, 0.44444444444444],
            [0.11270166537926, 0.88729833462074, 0.27777777777778]
        ], dtype=torch.float64, device=device)
    elif ch1 == 4:
        # 4-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.930568155797026, 0.069431844202974, 0.173927422568727],
            [0.669990521792428, 0.330009478207572, 0.326072577431273],
            [0.330009478207572, 0.669990521792428, 0.326072577431273],
            [0.069431844202974, 0.930568155797026, 0.173927422568727]
        ], dtype=torch.float64, device=device)
    elif ch1 == 5:
        # 5-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor( [
            [0.95308992295, 0.04691007705, 0.1184634425],
            [0.76923465505, 0.23076534495, 0.23931433525],
            [0.5, 0.5, 0.28444444445],
            [0.23076534495, 0.76923465505, 0.23931433525],
            [0.04691007705, 0.95308992295, 0.1184634425]
        ], dtype=torch.float64, device=device)
    elif ch1 == 6:
        # 6-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.966234757101576, 0.033765242898424, 0.085662246189585],
            [0.830604693233132, 0.169395306766868, 0.180380786524069],
            [0.619309593041598, 0.380690406958402, 0.233956967286345],
            [0.380690406958402, 0.619309593041598, 0.233956967286345],
            [0.169395306766868, 0.830604693233132, 0.180380786524069],
            [0.033765242898424, 0.966234757101576, 0.085662246189585]
        ], dtype=torch.float64, device=device)
    elif ch1 == 7:
        # 7-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.974553956171379, 0.025446043828621, 0.064742483084435],
            [0.870765592799697, 0.129234407200303, 0.139852695744638],
            [0.702922575688699, 0.297077424311301, 0.190915025252559],
            [0.5, 0.5, 0.208979591836735],
            [0.297077424311301, 0.702922575688699, 0.190915025252559],
            [0.129234407200303, 0.870765592799697, 0.139852695744638],
            [0.025446043828621, 0.974553956171379, 0.064742483084435]
        ], dtype=torch.float64, device=device)
    elif ch1 == 8:
        # 8-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.980144928248768, 0.019855071751232, 0.050614268145188],
            [0.898333238706813, 0.101666761293187, 0.111190517226687],
            [0.762766204958164, 0.237233795041836, 0.156853322938944],
            [0.591717321247825, 0.408282678752175, 0.181341891689181],
            [0.408282678752175, 0.591717321247825, 0.181341891689181],
            [0.237233795041836, 0.762766204958164, 0.156853322938944],
            [0.101666761293187, 0.898333238706813, 0.111190517226687],
            [0.019855071751232, 0.980144928248768, 0.050614268145188]
        ], dtype=torch.float64, device=device)
    elif ch1 == 9:
        # 9-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.98408011975400, 0.01591988024600, 0.04063719418080],
            [0.91801555366350, 0.08198444633650, 0.09032408034750],
            [0.80668571635050, 0.19331428364950, 0.13030534820150],
            [0.66212671170200, 0.33787328829800, 0.15617353852000],
            [0.5, 0.5, 0.16511967750050],
            [0.33787328829800, 0.66212671170200, 0.15617353852000],
            [0.19331428364950, 0.80668571635050, 0.13030534820150],
            [0.08198444633650, 0.91801555366350, 0.09032408034750],
            [0.01591988024600, 0.98408011975400, 0.04063719418080]
        ], dtype=torch.float64, device=device)
    elif ch1 == 10:
        # 10-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.986953264258586, 0.013046735741414, 0.033335672154344],
            [0.932531683344492, 0.067468316655508, 0.074725674575290],
            [0.839704784149512, 0.160295215850488, 0.109543181257991],
            [0.716697697064624, 0.283302302935376, 0.134633359654998],
            [0.574437169490816, 0.425562830509184, 0.147762112357376],
            [0.425562830509184, 0.574437169490816, 0.147762112357376],
            [0.283302302935376, 0.716697697064624, 0.134633359654998],
            [0.160295215850488, 0.839704784149512, 0.109543181257991],
            [0.067468316655508, 0.932531683344492, 0.074725674575290],
            [0.013046735741414, 0.986953264258586, 0.033335672154344]
        ], dtype=torch.float64, device=device)
    elif ch1 == 15:
        # 15-point Gaussian quadrature on a line segment
        GAUSSQUAD['LINE'] = torch.tensor([
            [0.006003740989757,  0.993996259010243, 0.030753241996117],
            [0.031363303799647,  0.968636696200353, 0.070366047488108],
            [0.075896708294786,  0.924103291705214, 0.107159220467172],
            [0.137791134319915,  0.862208865680085, 0.139570677926154],
            [0.214513913695731,  0.785486086304269, 0.166269205816994],
            [0.302924326461218,  0.697075673538782,  0.186161000015562],
            [0.399402953001283,  0.600597046998717,  0.198431485327112],
            [0.500000000000000,  0.500000000000000,  0.202578241925561],
            [0.600597046998717,  0.399402953001283,  0.198431485327112],
            [0.697075673538782,  0.302924326461218,  0.186161000015562],
            [0.785486086304269,  0.214513913695731,  0.166269205816994],
            [0.862208865680085,  0.137791134319915,  0.139570677926154],
            [0.924103291705214,  0.075896708294786,  0.107159220467172],
            [0.968636696200353,  0.031363303799647,  0.070366047488108],
            [0.993996259010243,  0.006003740989757,  0.030753241996117],
        ], dtype=torch.float64, device=device)
    else:
        GAUSSQUAD['LINE'] = []
        print('Wrong choice for line segments!')
        
    if ch2 == 1:
        # 1-point Gaussian quadrature on a rectangle 
        GAUSSQUAD['RECT'] = torch.tensor([
            [0.50000000000000, 0.50000000000000, 1.00000000000000]
        ], dtype=torch.float64, device=device)
    elif ch2 == 4:
        # 4-point Gaussian quadrature on a rectangle 
        GAUSSQUAD['RECT'] = torch.tensor([
             [0.788675134594813, 0.788675134594813, 0.250000000000000],
             [0.211324865405187, 0.788675134594813, 0.250000000000000],
             [0.788675134594813, 0.211324865405187, 0.250000000000000],
             [0.211324865405187, 0.211324865405187, 0.250000000000000]
        ], dtype=torch.float64, device=device)
    elif ch2 == 9:
        # 9-point Gaussian quadrature on a rectangle 
        GAUSSQUAD['RECT'] = torch.tensor([
            [0.887298334620740,   0.887298334620740, 0.077160493827162],
            [0.500000000000000,   0.887298334620740, 0.123456790123457],
            [0.112701665379260,   0.887298334620740, 0.077160493827162],
            [0.887298334620740,   0.500000000000000, 0.123456790123457],
            [0.500000000000000,   0.500000000000000, 0.197530864197527],
            [0.112701665379260,   0.500000000000000, 0.123456790123457],
            [0.887298334620740,   0.112701665379260, 0.077160493827162],
            [0.500000000000000,   0.112701665379260, 0.123456790123457],
            [0.112701665379260,   0.112701665379260, 0.077160493827162] 
        ], dtype=torch.float64, device=device)
    elif ch2 == 16:
        # 16-point Gaussian quadrature on a rectangle 
        GAUSSQUAD['RECT'] = torch.tensor([
            [0.930568155797026, 0.930568155797026, 0.030250748321401],
            [0.669990521792428, 0.930568155797026, 0.056712962962963],
            [0.330009478207572, 0.930568155797026, 0.056712962962963],
            [0.069431844202974, 0.930568155797026, 0.030250748321401],
            [0.930568155797026, 0.669990521792428, 0.056712962962963],
            [0.669990521792428, 0.669990521792428, 0.106323325752674],
            [0.330009478207572, 0.669990521792428, 0.106323325752674],
            [0.069431844202974, 0.669990521792428, 0.056712962962963],
            [0.930568155797026, 0.330009478207572, 0.056712962962963],
            [0.669990521792428, 0.330009478207572, 0.106323325752674],
            [0.330009478207572, 0.330009478207572, 0.106323325752674],
            [0.069431844202974, 0.330009478207572, 0.056712962962963],
            [0.930568155797026, 0.069431844202974, 0.030250748321401],
            [0.669990521792428, 0.069431844202974, 0.056712962962963],
            [0.330009478207572, 0.069431844202974, 0.056712962962963],
            [0.069431844202974, 0.069431844202974, 0.030250748321401]
        ], dtype=torch.float64, device=device)
    else:
        GAUSSQUAD['RECT'] = []
        print('Wrong choice!')
        
    if ch3 == 1:
        # 1-point (the center) quadrature for triangles, Exact for poly. deg.<=1
        GAUSSQUAD['TRIG'] = torch.tensor([
            [1/3, 1/3, 1/3, 1]
        ], dtype=torch.float64, device=device)
    elif ch3 == 3:
        # 3-point quadrature for triangles, Exact for poly. deg.<=2
        GAUSSQUAD['TRIG'] = torch.tensor([
            [4/6, 1/6, 1/6, 1/3],
            [1/6, 4/6, 1/6, 1/3],
            [1/6, 1/6, 4/6, 1/3]
        ], dtype=torch.float64, device=device)
    elif ch3 == 4:
        # 4-point quadrature for triangles, Exact for poly. deg.<=3
        a = 1/3
        b = 9/15
        c = 3/15
        u = -27/48
        v = 25/48
        GAUSSQUAD['TRIG'] = torch.tensor([
            [a, a, a, u],
            [b, c, c, v],
            [c, b, c, v],
            [c, c, b, v]
        ], dtype=torch.float64, device=device)
    elif ch3 == 6:
        # 6-point quadrature for triangles, Exact for poly. deg.<=4 
        a = 0.816847572980459
        b = 0.091576213509771
        c = 0.108103018168070
        d = 0.445948490915965
        u = 0.109951743655322  
        v = 0.223381589678011
        GAUSSQUAD['TRIG'] = torch.tensor([
           [a, b, b, u],
           [b, a, b, u],
           [b, b, a, u],
           [c, d, d, v],
           [d, c, d, v],
           [d, d, c, v],
        ], dtype=torch.float64, device=device)
    elif ch3 == 7:
        # 7-point quadrature for triangles, Exact for poly. deg.<=5
        a = 1/3;  
        b = (9+2*sqrt(15))/21  # b = 0.797426985353087; 
        c = (6-sqrt(15))/21    # c = 0.101286507323456; 
        d = (6+sqrt(15))/21    # d = 0.470142064105115; 
        e = (9-2*sqrt(15))/21  # e = 0.059715871789770;
        u = 0.225
        v = (155-sqrt(15))/1200.0 
        w = (155+sqrt(15))/1200;
        GAUSSQUAD['TRIG'] = torch.tensor([
           [a, a, a, u],
           [b, c, c, v],
           [c, b, c, v],
           [c, c, b, v],
           [d, d, e, w],
           [d, e, d, w],
           [e, d, d, w]
        ], dtype=torch.float64, device=device)
    else:
        GAUSSQUAD['TRIG'] = []
        print('Wrong choice for triangles!')
    
#     GAUSSQUAD = GAUSSQUAD.to(device)
    return GAUSSQUAD


def MeshGenerator_PureDiff(xa, xb, yc, yd, nx, ny, device):
    """
    Generates two triangular meshes for pure diffusion problems using PyTorch.

    Parameters:
        xa (float): Left boundary in x-direction.
        xb (float): Right boundary in x-direction.
        yc (float): Bottom boundary in y-direction.
        yd (float): Top boundary in y-direction.
        nx (int): Number of partitions in the x-direction.
        ny (int): Number of partitions in the y-direction.

    Returns:
        TriMeshes (list of TriMesh): List containing two TriMesh objects.
    """
    # Initialize the containers
    TriMeshes = [TriMesh(), TriMesh()]
    BndryDescMat = [None, None]
    BndryCondType = [None, None]
    
    id_boundary = [None, None]
    id_dirichlet = [None, None]

    FractureEgs = [None, None]
    DirichletEgs = [None, None]
    NeumannEgs = [None, None]
    EqnBC_Glob = [None, None]
    DOFs = [None, None]

    x_start = [None, None]
    x_end = [None, None]
    y_start = [None, None]
    y_end = [None, None]

    # Calculating delta_x and setting start/end points for x and y
    delta_x = (xb - xa) / nx
    x_start[0], x_start[1] = xa, (nx // 2) * delta_x
    x_end[0], x_end[1] = x_start[1], xb
    y_start[0], y_start[1] = yc, yc
    y_end[0], y_end[1] = yd, yd

    # Mesh generation
    for k in range(2):
        # Ensure that nx // 2 is at least 1 to avoid zero partitions
        partitions_x = max(nx // 2, 1)
        TriMeshes[k].RectDom_TriMesh_GenUnfm(
            x_start[k], x_end[k], partitions_x, y_start[k], y_end[k], ny, device, status=2
        )

    # Boundary description matrix
    for k in range(2):
        BndryDescMat[k] = torch.tensor([
            [x_start[k], y_start[k], x_end[k], y_start[k], 0, -1],
            [x_end[k], y_start[k], x_end[k], y_end[k], 1, 0],
            [x_end[k], y_end[k], x_start[k], y_end[k], 0, 1],
            [x_start[k], y_end[k], x_start[k], y_start[k], -1, 0]
        ], dtype=torch.float64, device=device)

    # Boundary condition types
    BndryCondType[0] = torch.tensor([0, 2, 1, 2, 1], dtype=torch.int64, device=device)
    BndryCondType[1] = torch.tensor([0, 2, 1, 2, 1], dtype=torch.int64, device=device)

    # Degree of Freedom (DOFs)
    for k in range(2):
        DOFs[k] = TriMeshes[k].NumEms+TriMeshes[k].NumEgs
#     print(TriMeshes[0].DOFs)
    # Enrich the TriMesh
    for k in range(2):
        TriMeshes[k].TriMesh_Enrich2(BndryDescMat[k], device)
        TriMeshes[k].TriMesh_Enrich3(BndryDescMat[k], device)

    # Equation boundary conditions
    EqnBC_Glob[0] = EqnBC()
    EqnBC_Glob[1] = EqnBC()

    # Setting permeability
    GAUSSQUAD = SetGaussQuad(2, 4, 3, device)  # Ensure SetGaussQuad is translated to PyTorch
    PermK = [None, None]
    for k in range(2):
        Eqn_Sub = EqnBC_Glob[k]
        PermK[k] = Darcy_SmplnPerm_TriMesh(Eqn_Sub.fxnK, TriMeshes[k], GAUSSQUAD)

    # Assign properties to TriMeshes
    for k in range(2):
        TriMeshes[k].BndryDescMat = BndryDescMat[k]
        TriMeshes[k].EqnBC_Glob = EqnBC_Glob[k]
        TriMeshes[k].DOFs = DOFs[k]
        TriMeshes[k].PermK = PermK[k]
        TriMeshes[k].BndryCondType = BndryCondType[k]

    # Boundary and Dirichlet edge settings
    id_boundary[0], id_boundary[1] = 2, 4
    id_dirichlet[0], id_dirichlet[1] = 4, 2

    # Find fracture edges
    for k in range(2):
        # Using list comprehensions is less efficient; using torch.nonzero with masking
        mask = TriMeshes[k].BndryEdge == id_boundary[k]
        FractureEgs[k] = torch.nonzero(mask, as_tuple=False).squeeze()
        FractureEgs[k] = FractureEgs[k].to(device)
        
    for k in range(2):
        TriMeshes[k].FractureEgs = FractureEgs[k].unsqueeze(0)  # Shape: (1, NumFractureEgs)
#     print(TriMeshes[0].FractureEgs.shape)
    # Nodes and elements
    edges = [TriMeshes[0].edge, TriMeshes[1].edge]
    nodes = [TriMeshes[0].node, TriMeshes[1].node]
    elems = [TriMeshes[0].elem, TriMeshes[1].elem]

    # Find Dirichlet edges
    for k in range(2):
        # Condition: BndryEdge == id_dirichlet[k] AND node's y-coordinate <= 0.2
        condition = (TriMeshes[k].BndryEdge == id_dirichlet[k])
        DirichletEgs[k] = torch.nonzero(condition, as_tuple=False).squeeze()
        DirichletEgs[k] = DirichletEgs[k].to(device)
        
    # Find Neumann edges
    for k in range(2):
        # Condition: BndryEdge == 3 OR BndryEdge == 1 OR (BndryEdge == id_dirichlet[k] AND node's y-coordinate > 0.2)
        condition = (
            (TriMeshes[k].BndryEdge == 3) |
            (TriMeshes[k].BndryEdge == 1) 
        )
        NeumannEgs[k] = torch.nonzero(condition, as_tuple=False).squeeze()
        NeumannEgs[k] = NeumannEgs[k].to(device)
        
    # Assign Dirichlet and Neumann edges
    for k in range(2):
        TriMeshes[k].DirichletEgs = DirichletEgs[k].unsqueeze(0)  # Shape: (1, NumDirichletEgs)
        TriMeshes[k].NeumannEgs = NeumannEgs[k].unsqueeze(0)      # Shape: (1, NumNeumannEgs)

    return TriMeshes


class EqnBC_frt:
    def __init__(self):
        self.fxnf = self.fxnf_frt

    def fxnf_frt(self, pt, t):
        """
        Function f defined over points pt at time t for fractures.

        Parameters:
            pt (torch.Tensor): Tensor of points (NumPts x 2)
            t (float): Time variable

        Returns:
            f (torch.Tensor): Evaluated function values (NumPts,)
        """
#         x = pt[:, 0]
        y = pt[:, 0]

        # Uncomment one of the following lines to define f as needed

        # Example 1:
        # f = (3 * math.pi**2) * torch.exp(math.pi**2 * t) * torch.sin(math.pi * x) * torch.sin(math.pi * y)

        # Example 2:
        # f = torch.exp(t) * x * (x - 1) * y * (y - 1) - 2 * torch.exp(t) * (x * (x - 1) + y * (y - 1))

        # Default (zero function):
        f = torch.zeros_like(y[:, None], device=pt.device)

        # Example 3:
        # f = (3 * math.pi**2) * torch.exp(math.pi**2 * t) * torch.cos(math.pi * x) * torch.cos(math.pi * y)

        return f
    
def compute_diffusion_single(diff_1subdom, diff_2subdom, TriMesh, device):
    diffusion_subdom = [None] * 2
    for k1 in range(2):
        mesh = TriMesh[k1]
#         print(mesh.NumEms)
        diffusion_subdom[k1] =torch.zeros((mesh.NumEms, 1), device = device)

    diffusion_subdom[0][:, 0] = diff_1subdom
    diffusion_subdom[1][:, 0] = diff_2subdom

    return diffusion_subdom

def Darcy_MFEM_StiffMat(TriMesh, dt, diffusion, poros_subdom, device):
    """
    Assemble the global stiffness matrix for the Darcy equation using MFEM and PyTorch.

    Parameters:
        TriMesh (TriMesh): The mesh object containing mesh information.
        dt (float): Time step or similar scalar.
        diffusion (torch.tensor or callable): Diffusion coefficient(s).
        poros_subdom (torch.tensor or float): Porosity for subdomains.

    Returns:
        GlbMat (torch.sparse.FloatTensor): The assembled global stiffness matrix.
    """
    
    # Mesh information
    NumEms = TriMesh.NumEms
    NumEgs = TriMesh.NumEgs
    area = TriMesh.area          # Shape: (NumEms,)
    LenEg = TriMesh.LenEg        # Shape: (NumEgs,)
    sn = TriMesh.SignEmEg        # Shape: (NumEms, 3)

    # Compute GMRT0K1 using the helper function
    GMRT0K1 = Hdiv_TriRT0_EgBas_LocalMat(TriMesh, diffusion, device)  # Shape: (NumEgs, 3, 3)
#     print(GMRT0K1.shape)
#     print(GMRT0K1)
    # Initialize MatA as a sparse tensor for efficient construction
    indices = []
    values = []

    # Assemble MatA
    for i in range(3):
        II = TriMesh.elem2edge[:, i]  # Edge indices for the i-th edge of each element
        for j in range(3):
            JJ = TriMesh.elem2edge[:, j]  # Edge indices for the j-th edge of each element

            # Element-wise multiplication
            data = GMRT0K1[:, i, j] * sn[:, i] * sn[:, j]

            # Collect indices and values for constructing the sparse matrix
            indices.append(torch.stack([II, JJ]))
            values.append(data)

    # Convert to tensors and stack them
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)

    # Create sparse tensor MatA
    MatA = torch.sparse_coo_tensor(indices, values, size=(NumEgs, NumEgs))
    MatA = MatA.to(TriMesh.node.device)
    # Initialize MatB as a sparse tensor
    indices_B = []
    values_B = []
    II = torch.arange(NumEms, device=TriMesh.node.device)  # Element indices
    for j in range(3):
        JJ = TriMesh.elem2edge[:, j]  # Edge indices for the j-th edge of each element
        data = -LenEg[TriMesh.elem2edge[:, j]] * sn[:, j]  # Element-wise multiplication

        # Collect indices and values for constructing the sparse matrix
        indices_B.append(torch.stack([II, JJ]))
        values_B.append(data)

    # Convert to tensors and stack them
    indices_B = torch.cat(indices_B, dim=1)
    values_B = torch.cat(values_B)
    
    # Create sparse tensor MatB
    MatB = torch.sparse_coo_tensor(indices_B, values_B, size=(NumEms, NumEgs), dtype=torch.float64, device=TriMesh.node.device)
#     MatB = MatB.to(TriMesh.node.device)
    # Assemble MatC using porosity and element areas
    if isinstance(poros_subdom, (float, int)):
        # If poros_subdom is a scalar, multiply it directly
        MatC = poros_subdom * torch.diag(area)
    else:
        # If poros_subdom is a tensor, perform element-wise multiplication
        MatC = torch.diag(poros_subdom * area)
    MatC = MatC.to(TriMesh.node.device)
    
    # Assemble the global matrix GlbMat using block matrix construction
    # GlbMat = [MatA, MatB'; dt*MatB, -MatC]
    top = torch.cat([MatA.to_dense(), MatB.transpose(0, 1).to_dense()], dim=1)
    bottom = torch.cat([dt * MatB.to_dense(), -MatC], dim=1)
    GlbMat = torch.cat([top, bottom], dim=0)

    # Convert GlbMat back to sparse format if needed (dense by default in PyTorch)
#     GlbMat = GlbMat.to_sparse()

    return GlbMat

def ComputeAblock(TriMesh, dt, diffusion, device):
    NumEms = TriMesh.NumEms
    NumEgs = TriMesh.NumEgs
    area = TriMesh.area          # Shape: (NumEms,)
    LenEg = TriMesh.LenEg        # Shape: (NumEgs,)
    sn = TriMesh.SignEmEg        # Shape: (NumEms, 3)
    
    GMRT0K1 = Hdiv_TriRT0_EgBas_LocalMat_Vectorized(TriMesh, diffusion, device)
    # Compute GMRT0K1 using the helper function
#     GMRT0K1 = Hdiv_TriRT0_EgBas_LocalMat(TriMesh, diffusion, device)  # Shape: (NumEgs, 3, 3)
#     print(GMRT0K1)
    # Initialize MatA as a sparse tensor for efficient construction
    indices = []
    values = []
#     print(GMRT0K1.shape)
    # Assemble MatA
    for i in range(3):
        II = TriMesh.elem2edge[:, i]  # Edge indices for the i-th edge of each element
        for j in range(3):
            JJ = TriMesh.elem2edge[:, j]  # Edge indices for the j-th edge of each element

            # Element-wise multiplication
            data = GMRT0K1[:, i, j] * sn[:, i] * sn[:, j]

            # Collect indices and values for constructing the sparse matrix
            indices.append(torch.stack([II, JJ]))
            values.append(data)

    # Convert to tensors and stack them
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)

    # Create sparse tensor MatA
    MatA = torch.sparse_coo_tensor(indices, values, size=(NumEgs, NumEgs))
    MatA = MatA.to(TriMesh.node.device)
    return MatA

def ComputeBandCblocks(TriMesh, dt, poros_subdom,  device):
    NumEms = TriMesh.NumEms
    NumEgs = TriMesh.NumEgs
    area = TriMesh.area          # Shape: (NumEms,)
    LenEg = TriMesh.LenEg        # Shape: (NumEgs,)
    sn = TriMesh.SignEmEg 
    indices_B = []
    values_B = []
    II = torch.arange(NumEms, device=TriMesh.node.device)  # Element indices
    for j in range(3):
        JJ = TriMesh.elem2edge[:, j]  # Edge indices for the j-th edge of each element
        data = -LenEg[TriMesh.elem2edge[:, j]] * sn[:, j]  # Element-wise multiplication

        # Collect indices and values for constructing the sparse matrix
        indices_B.append(torch.stack([II, JJ]))
        values_B.append(data)

    # Convert to tensors and stack them
    indices_B = torch.cat(indices_B, dim=1)
    values_B = torch.cat(values_B)
    
    # Create sparse tensor MatB
    MatB = torch.sparse_coo_tensor(indices_B, values_B, size=(NumEms, NumEgs), dtype=torch.float64, device=TriMesh.node.device)
#     MatB = MatB.to(TriMesh.node.device)
    # Assemble MatC using porosity and element areas
    if isinstance(poros_subdom, (float, int)):
        # If poros_subdom is a scalar, multiply it directly
        MatC = poros_subdom * torch.diag(area)
    else:
        # If poros_subdom is a tensor, perform element-wise multiplication
        MatC = torch.diag(poros_subdom * area)
    MatC = MatC.to(TriMesh.node.device)
    return MatB, MatC

def Darcy_MFEM_StiffMat_Modify(A, B, C, dt):
    top = torch.cat([A.to_dense(), B.transpose(0, 1).to_dense()], dim=1)
    bottom = torch.cat([dt * B.to_dense(), -C], dim=1)
    GlbMat = torch.cat([top, bottom], dim=0)
    return GlbMat

def Hdiv_TriRT0_EgBas_LocalMat_Vectorized(TriMesh, diffusion, device):
    """
    Compute the local Gram matrices for individual elements using PyTorch.
    """
    # Mesh info
    NumEms = TriMesh.NumEms
    e2g = TriMesh.elem2edge
    area = TriMesh.area
    LenEg = TriMesh.LenEg

    # Gram matrices: Locally for individual elements
    GMK = torch.zeros((NumEms, 3, 3), dtype=torch.float64, device=device)
    GM = torch.tensor([[2, 0, 1, 0, 1, 0],
                       [0, 2, 0, 1, 0, 1],
                       [1, 0, 2, 0, 1, 0],
                       [0, 1, 0, 2, 0, 1],
                       [1, 0, 1, 0, 2, 0],
                       [0, 1, 0, 1, 0, 2]], dtype=torch.float64, device=device)

    # Compute L matrices: Use broadcasting to construct batched diagonal matrices
#     L = (1 / diffusion).unsqueeze(-1).unsqueeze(-1) * torch.eye(6, dtype=torch.float64, device=device).unsqueeze(0)  # Shape: (NumEms, 6, 6)
    L = (1 / diffusion).unsqueeze(-1) * torch.eye(6, dtype=torch.float64, device=device)
#     print(L.shape)
    # Compute A matrices in batch
    A = torch.matmul(GM.unsqueeze(0), L)  # Shape: (NumEms, 6, 6)
 
    # Gather vertices for all elements
    vrtx = TriMesh.node[TriMesh.elem[:, :3], :]  # Shape: (NumEms, 3, 2)
#     vrtx_flat = vrtx.transpose(1, 2).reshape(NumEms, -1, 1)  # Shape: (NumEms, 6, 1)  
#     print(vrtx.permute(0, 2, 1))
    vrtx_flat = vrtx.reshape(NumEms, 6, 1)  # Shape: (NumEms, 6)
#     vrtx_flat = vrtx_flat.unsqueeze(-1)  # Shape: (NumEms, 6, 1)
#     print(vrtx_flat)
    # Expand vrtx for B2 computation
    vrtx_T = vrtx.permute(0, 2, 1)
    vrtx_Tstack = torch.cat([vrtx_T, vrtx_T, vrtx_T], dim=1)  # Shape: (NumEms, 6, 2)

    # Compute B1 and B2
    B1 = torch.cat([vrtx_flat, vrtx_flat, vrtx_flat], dim=2)  # Shape: (NumEms, 6, 3)
#     print(B1)
#     print('line')
    B2 = vrtx_Tstack
#     B2 = vrtx_repeated.permute(0, 2, 1)  # Shape: (NumEms, 6, 3)
#     print(B2)
    # Compute B
    B = (B1 - B2).to(torch.float64)  # Shape: (NumEms, 6, 3)
    # Create B matrices in batch
    edge_lengths = LenEg[e2g[:, :3]]  # Shape: (NumEms, 3)
    C = torch.diag_embed(edge_lengths).to(torch.float64)  # Shape: (NumEms, 3, 3)
#     print(A.shape)
#     print(B.shape)
#     print(C.shape)
    # Compute the local Gram matrices GMK in batch
    area = area.to(torch.float64).unsqueeze(-1).unsqueeze(-1)  # Shape: (NumEms, 1, 1)
    GMK = (1 / 48) / area * torch.matmul(C.transpose(-1, -2), torch.matmul(B.transpose(-1, -2),\
                                                                           torch.matmul(A, torch.matmul(B, C))))  # Shape: (NumEms, 3, 3)

#     for ie in range(NumEms):
#         # Compute L matrix (1/diffusion for the element)
#         L = (1/diffusion[ie]) * torch.eye(6, dtype=torch.float64, device=device)
        
#         # Matrix A
#         A = torch.matmul(GM, L)
#         A = A.to(torch.float64)
# #         print(A)
#         # Extract the vertices for the element
#         vrtx = TriMesh.node[TriMesh.elem[ie, :3], :].T  # Transpose to match the (2, 3) structure

#         vrtx_flat = vrtx.t().reshape(-1, 1)
#         # Create matrix B
#         B1 = torch.column_stack([vrtx_flat, vrtx_flat, vrtx_flat])
#         B2 = torch.cat([vrtx, vrtx, vrtx], dim=0)
#         B = B1 - B2
#         B = B.to(torch.float64)

#         # Diagonal matrix C using edge lengths
#         C = torch.diag(LenEg[e2g[ie, :3]])
#         C = C.to(torch.float64)
#         C = C.to(device)
#         area = area.to(torch.float64)
#         # Compute the local Gram matrix GMK
#         GMK[ie, :, :] = (1/48) / area[ie] * torch.matmul(C.T, torch.matmul(B.T, torch.matmul(A, torch.matmul(B, C))))

    return GMK
def Hdiv_TriRT0_EgBas_LocalMat(TriMesh, diffusion, device):
    """
    Compute the local Gram matrices for individual elements using PyTorch.
    """
    # Mesh info
    NumEms = TriMesh.NumEms
    e2g = TriMesh.elem2edge
    area = TriMesh.area
    LenEg = TriMesh.LenEg

    # Gram matrices: Locally for individual elements
    GMK = torch.zeros((NumEms, 3, 3), dtype=torch.float64, device=device)
    GM = torch.tensor([[2, 0, 1, 0, 1, 0],
                       [0, 2, 0, 1, 0, 1],
                       [1, 0, 2, 0, 1, 0],
                       [0, 1, 0, 2, 0, 1],
                       [1, 0, 1, 0, 2, 0],
                       [0, 1, 0, 1, 0, 2]], dtype=torch.float64, device=device)

    for ie in range(NumEms):
        # Compute L matrix (1/diffusion for the element)
        L = (1/diffusion[ie]) * torch.eye(6, dtype=torch.float64, device=device)
        
        # Matrix A
        A = torch.matmul(GM, L)
        A = A.to(torch.float64)
#         print(A)
        # Extract the vertices for the element
        vrtx = TriMesh.node[TriMesh.elem[ie, :3], :].T  # Transpose to match the (2, 3) structure

        vrtx_flat = vrtx.t().reshape(-1, 1)
        # Create matrix B
        B1 = torch.column_stack([vrtx_flat, vrtx_flat, vrtx_flat])
        B2 = torch.cat([vrtx, vrtx, vrtx], dim=0)
        B = B1 - B2
        B = B.to(torch.float64)

        # Diagonal matrix C using edge lengths
        C = torch.diag(LenEg[e2g[ie, :3]])
        C = C.to(torch.float64)
        C = C.to(device)
        area = area.to(torch.float64)
        # Compute the local Gram matrix GMK
        GMK[ie, :, :] = (1/48) / area[ie] * torch.matmul(C.T, torch.matmul(B.T, torch.matmul(A, torch.matmul(B, C))))

    return GMK

def ComputeDblocks(TriMesh, h, dt, ny, device):
    num_frt_egs = [TriMesh[k].FractureEgs.shape[1] for k in range(2)]

    # Initialize Com_matrix as a list of two sparse matrices
    Com_matrix = [torch.zeros((TriMesh[k].NumEgs, ny), dtype=torch.float64, device=device) for k in range(2)]

    # Assemble Com_matrix
    for k in range(2):
        frt_egs = TriMesh[k].FractureEgs.flatten()
        num_frt = num_frt_egs[k]

        if num_frt > ny:
            raise ValueError(f"Number of fracture edges {num_frt} exceeds ny={ny}.")

        # Assign h to (frt_egs[i], i) for i in range(num_frt)
        rows = frt_egs
        cols = torch.arange(num_frt)
        data = h * torch.ones(num_frt, dtype=torch.float64, device=device)
        
        for i in range(len(rows)):
            Com_matrix[k][rows[i], cols[i]] += data[i]
        # Convert to CSR-like for efficient ops
        Com_matrix[k] = Com_matrix[k].to_sparse()
    return Com_matrix

def Assemble_StiffMax_PureDiff_Modify(TriMesh, h, node_frt, ny, Glb_Mat, MatD, alpha_frt, dt, poros_frt, device):
    """
    Assemble the global system matrix for the Darcy equation with fractures using PyTorch.
    
    Parameters:
        TriMesh (list): List of two TriMesh objects.
        h (float): Spatial step or similar scalar.
        node_frt (int): Number of fracture nodes.
        ny (int): Number of y-direction nodes or similar.
        Glb_Mat (list): List of two global matrices, Glb_Mat[0] and Glb_Mat[1].
        alpha_frt (float): Scalar parameter.
        dt (float): Time step or similar scalar.
        poros_frt (float): Porosity for fractures.

    Returns:
        System_Mat (torch.sparse_coo_tensor): The assembled global system matrix.
    """
    
    # Initialize Adiag
    Adiag = torch.zeros(node_frt, device=device)
    Adiag[0] = 2
    Adiag[-1] = 2
    Adiag[1:ny] = 4  # Assign 4 to indices 1 to ny-1

    # Initialize Alow and Aup
    Alow = torch.ones(node_frt, device=device)
    Aup = torch.ones(node_frt, device=device)

    # Create A_frt as a sparse tridiagonal matrix
    row = torch.cat([torch.arange(1, node_frt), torch.arange(node_frt), torch.arange(node_frt-1)])
    col = torch.cat([torch.arange(node_frt-1), torch.arange(node_frt), torch.arange(1, node_frt)])
    data = torch.cat([Alow[1:], Adiag, Aup[:-1]])
    A_frt = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),  # Indices need to be stacked
            values=(h / 6) * data,
            size=(node_frt, node_frt),
            dtype=torch.float64,  # Set appropriate data type
            device=device
        )

    # Create B_frt as a sparse matrix with diagonals at -1 and 0
    Bdiag = torch.ones(ny, device=device)  # Main diagonal
    Blow = -1*torch.ones(ny, device=device)  # Lower diagonal (only ny-1 elements)

    # Create row and column indices for the sparse matrix
    row_indices = torch.cat([torch.arange(ny), torch.arange(1, ny+1)])  # Main diagonal and lower diagonal
    col_indices = torch.cat([torch.arange(ny), torch.arange(ny)])   # Column indices for main and lower diagonal

    # Combine row and column indices into a single tensor
    indices = torch.stack([row_indices, col_indices])

    # Combine data for the sparse tensor
    data = torch.cat([Bdiag, Blow])  # Main diagonal and lower diagonal values

    # Create the sparse COO tensor
    B_frt = torch.sparse_coo_tensor(indices, data, size=(node_frt, ny), dtype=torch.float64, device=device)
    # Create the sparse COO tensor
    # Create C_frt as a diagonal sparse matrix
    C_frt = poros_frt*h*torch.sparse_coo_tensor(torch.stack([torch.arange(ny), torch.arange(ny)]),\
                                                torch.ones(ny), size=(ny, ny), dtype=torch.float64, device=device)

    # Create Glb_Mat_frt as a block matrix
    A_frt_inv = (alpha_frt)**-1 * A_frt
    
    
    # Use torch.block_diag on dense tensors
    row1 = torch.cat((A_frt_inv, B_frt), dim=1)

    # Second row: [dt * B_frt.transpose(), -C_frt]
    row2 = torch.cat((dt * B_frt.transpose(0, 1), -C_frt), dim=1)

    # Combine both rows into a block matrix
    Glb_Mat_frt = torch.cat((row1, row2), dim=0)
    Glb_Mat_frt = Glb_Mat_frt.to_dense()

    total_size = TriMesh[0].DOFs + TriMesh[1].DOFs + 2 * ny + 1
    System_Mat = torch.zeros((total_size, total_size), dtype=torch.float64, device=device)  # PyTorch has no LIL matrix, so use dense

    # Assign Glb_Mat to blocks
    block1_start = 0
    block1_end = TriMesh[0].DOFs
    block2_start = block1_end
    block2_end = block2_start + TriMesh[1].DOFs
    block_frt_start = block2_end
    block_frt_end = block_frt_start + (2 * ny + 1)

    # Use slicing for assignments
    System_Mat[block1_start:block1_end, block1_start:block1_end] = Glb_Mat[0]
    System_Mat[block2_start:block2_end, block2_start:block2_end] = Glb_Mat[1]
    System_Mat[block_frt_start:block_frt_end, block_frt_start:block_frt_end] = Glb_Mat_frt
    # Compute number of fracture edges for each mesh
    num_frt_egs = [TriMesh[k].FractureEgs.shape[1] for k in range(2)]

    # Assign Com_matrix to System_Mat
    row_start_1 = 0
    row_end_1 = TriMesh[0].NumEgs
    col_start_1 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_1 = col_start_1 + ny
    System_Mat[row_start_1:row_end_1, col_start_1:col_end_1] = MatD[0].to_dense()
    System_Mat[col_start_1:col_end_1, row_start_1:row_end_1] = dt * MatD[0].transpose(0, 1).to_dense()

    row_start_2 = TriMesh[0].DOFs
    row_end_2 = TriMesh[0].DOFs + TriMesh[1].NumEgs
    col_start_2 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_2 = col_start_2 + ny
    
#     print([row_start_2, row_end_2, col_start_2, col_end_2])
    System_Mat[row_start_2:row_end_2, col_start_2:col_end_2] = MatD[1].to_dense()
    System_Mat[col_start_2:col_end_2, row_start_2:row_end_2] = dt*MatD[1].transpose(0, 1).to_dense()
#     print(System_Mat[col_start_2:col_end_2, row_start_2:row_end_2].to_sparse())
    # Convert System_Mat to a sparse tensor
    System_Mat = System_Mat.to_sparse()
#     System_Mat = System_Mat.to(torch.float64)
    return System_Mat

def StiffMax_PureDiff_Vectorized_General(TriMesh, h, node_frt, ny, Glb_Mat, alp_frt_vec, keppa, dt, poros_frt, device):
    """
    Assemble the global system matrix for the Darcy equation with fractures using PyTorch.
    
    Parameters:
        TriMesh (list): List of two TriMesh objects.
        h (float): Spatial step or similar scalar.
        node_frt (int): Number of fracture nodes.
        ny (int): Number of y-direction nodes or similar.
        Glb_Mat (list): List of two global matrices, Glb_Mat[0] and Glb_Mat[1].
        alpha_frt (float): Scalar parameter.
        dt (float): Time step or similar scalar.
        poros_frt (float): Porosity for fractures.

    Returns:
        System_Mat (torch.sparse_coo_tensor): The assembled global system matrix.
    """
    
    # Initialize Adiag
    Adiag = torch.zeros(node_frt, device=device, dtype=torch.float64)
    Adown = torch.zeros(ny, device=device, dtype=torch.float64)
    Aup = torch.zeros(ny, device=device, dtype=torch.float64)

    # Populate Adiag for boundary values
    Adiag[0] = 2 / alp_frt_vec[0]
    Adiag[-1] = 2 / alp_frt_vec[-1]

    # Populate Adiag for inner values
    for i in range(1, ny):
        Adiag[i] = 2 / alp_frt_vec[i - 1] + 2 / alp_frt_vec[i]

    # Populate Adown and Aup
    Adown[:] = 1 / alp_frt_vec[:]
    Aup[:] = 1 / alp_frt_vec[:]

    # Combine the diagonals for sparse matrix creation
    row_indices = torch.cat([torch.arange(1, node_frt, device=device), torch.arange(node_frt, device=device),\
                             torch.arange(node_frt - 1, device=device)])
    col_indices = torch.cat([torch.arange(node_frt-1, device=device), torch.arange(node_frt, device=device),\
                             torch.arange(1, node_frt, device=device)])
    data_values = torch.cat([Adown, Adiag, Aup])

    # Create the sparse matrix A_frt
    A_frt = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=(h/ 6) * data_values,
        size=(node_frt, node_frt),
        dtype=torch.float64,
        device=device
    )

    # Create B_frt as a sparse matrix with diagonals at -1 and 0
    Bdiag = torch.ones(ny, device=device)  # Main diagonal
    Blow = -1*torch.ones(ny, device=device)  # Lower diagonal (only ny-1 elements)

    # Create row and column indices for the sparse matrix
    row_indices = torch.cat([torch.arange(ny), torch.arange(1, ny+1)])  # Main diagonal and lower diagonal
    col_indices = torch.cat([torch.arange(ny), torch.arange(ny)])   # Column indices for main and lower diagonal

    # Combine row and column indices into a single tensor
    indices = torch.stack([row_indices, col_indices])

    # Combine data for the sparse tensor
    data = torch.cat([Bdiag, Blow])  # Main diagonal and lower diagonal values

    # Create the sparse COO tensor
    B_frt = torch.sparse_coo_tensor(indices, data, size=(node_frt, ny), dtype=torch.float64, device=device)
    # Create the sparse COO tensor
    C_frt = poros_frt*h*torch.sparse_coo_tensor(torch.stack([torch.arange(ny), torch.arange(ny)]),\
                                                torch.ones(ny), size=(ny, ny), dtype=torch.float64, device=device)

    # Create Glb_Mat_frt as a block matrix        

    # Use torch.block_diag on dense tensors
    row1 = torch.cat((A_frt, B_frt), dim=1)

    # Second row: [dt * B_frt.transpose(), -C_frt]
    row2 = torch.cat((dt * B_frt.transpose(0, 1), -C_frt), dim=1)

    # Combine both rows into a block matrix
    Glb_Mat_frt = torch.cat((row1, row2), dim=0)
    Glb_Mat_frt = Glb_Mat_frt.to_dense()

    total_size = TriMesh[0].DOFs + TriMesh[1].DOFs + 2 * ny + 1
    System_Mat = torch.zeros((total_size, total_size), dtype=torch.float64, device=device)  # PyTorch has no LIL matrix, so use dense
    xi = 1
    
    main_diag = [None]*2
    for k in range(2):
        main_diag[k] = torch.zeros(TriMesh[k].DOFs, dtype=torch.float64)

    # Update main diagonal for each TriMesh
    for k in range(2):
        main_diag[k][TriMesh[k].FractureEgs] += h / keppa[:]

    # Construct added matrices
    AddedMat = [None]*2
    for k in range(2):
        # Create sparse diagonal matrix using scipy's spdiags
        diag_values = xi * main_diag[k]  # Convert to numpy for scipy
        AddedMat[k] = torch.diag_embed(diag_values)

    # Update global matrices
    for k in range(2):
        Glb_Mat[k] += AddedMat[k]
        
    # Assign Glb_Mat to blocks
    System_Mat[0:TriMesh[0].DOFs, 0:TriMesh[1].DOFs] = Glb_Mat[0]
    System_Mat[TriMesh[0].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs,
               TriMesh[0].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs] = Glb_Mat[1]
    System_Mat[TriMesh[0].DOFs + TriMesh[1].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs + (2 * ny + 1),
               TriMesh[0].DOFs + TriMesh[1].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs + (2 * ny + 1)] = Glb_Mat_frt

    # Compute number of fracture edges for each mesh
    num_frt_egs = [TriMesh[k].FractureEgs.shape[1] for k in range(2)]

    # Initialize Com_matrix as a list of two sparse matrices
    Com_matrix = [torch.zeros((TriMesh[k].NumEgs, ny), dtype=torch.float64, device=device) for k in range(2)]
    Com_matrix_subdom = [torch.zeros((TriMesh[k].NumEgs, TriMesh[1-k].NumEgs), dtype=torch.float64, device=device) for k in range(2)]
    
    # Assemble Com_matrix
    for k in range(2):
        frt_egs = TriMesh[k].FractureEgs.flatten()
        num_frt = num_frt_egs[k]

        if num_frt > ny:
            raise ValueError(f"Number of fracture edges {num_frt} exceeds ny={ny}.")

        # Assign h to (frt_egs[i], i) for i in range(num_frt)
        rows = frt_egs
        cols = torch.arange(num_frt)
        data = h * torch.ones(num_frt, dtype=torch.float64, device=device)
        
        for i in range(len(rows)):
            Com_matrix[k][rows[i], cols[i]] += data[i]
        # Convert to CSR-like for efficient ops
        Com_matrix[k] = Com_matrix[k].to_sparse()
        
    for k in range(2):
        # Get the current main_diag for 3-k (Python uses 0-based indexing, so subtract 1 from MATLAB's 3-k)
        main_diag_values = (1 - xi) * main_diag[1 - k]  # 1 - k corresponds to MATLAB's 3 - k

        # Create a diagonal matrix
        diag_matrix = torch.diag(main_diag_values[0:TriMesh[k].NumEgs])

        # Add to Com_matrix_subdom[k]
        Com_matrix_subdom[k] += diag_matrix
        
    # Assign Com_matrix to System_Mat
    row_start_1 = 0
    row_end_1 = TriMesh[0].NumEgs
    col_start_1 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_1 = col_start_1 + ny
    System_Mat[row_start_1:row_end_1, col_start_1:col_end_1] = Com_matrix[0].to_dense()
    System_Mat[col_start_1:col_end_1, row_start_1:row_end_1] = dt * Com_matrix[0].transpose(0, 1).to_dense()

    row_start_2 = TriMesh[0].DOFs
    row_end_2 = TriMesh[0].DOFs + TriMesh[1].NumEgs
    col_start_2 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_2 = col_start_2 + ny
    
#     print([row_start_2, row_end_2, col_start_2, col_end_2])
    System_Mat[row_start_2:row_end_2, col_start_2:col_end_2] = Com_matrix[1].to_dense()
    System_Mat[col_start_2:col_end_2, row_start_2:row_end_2] = dt*Com_matrix[1].transpose(0, 1).to_dense()
    
    row_start_3 = 0
    row_end_3 = TriMesh[0].NumEgs
    col_start_3 = TriMesh[0].DOFs 
    col_end_3 = col_start_3 + TriMesh[1].NumEgs
    
    System_Mat[row_start_3:row_end_3, col_start_3:col_end_3] -=Com_matrix_subdom[0]
    
    row_start_4 = TriMesh[0].DOFs 
    row_end_4 = TriMesh[1].NumEgs+TriMesh[0].DOFs 
    col_start_4 = 0
    col_end_4 = TriMesh[0].NumEgs
    
    System_Mat[row_start_4:row_end_4, col_start_4:col_end_4] -=Com_matrix_subdom[1]

#     print(System_Mat[col_start_2:col_end_2, row_start_2:row_end_2].to_sparse())
    # Convert System_Mat to a sparse tensor
    System_Mat = System_Mat.to_sparse()
#     System_Mat = System_Mat.to(torch.float64)
    return System_Mat

def Assemble_StiffMax_PureDiff_Vectorized(TriMesh, h, node_frt, ny, Glb_Mat1st, Glb_Mat2nd, alpha_frt, dt, poros_frt, device):
    """
    Assemble the global system matrix for the Darcy equation with fractures using PyTorch.
    
    Parameters:
        TriMesh (list): List of two TriMesh objects.
        h (float): Spatial step or similar scalar.
        node_frt (int): Number of fracture nodes.
        ny (int): Number of y-direction nodes or similar.
        Glb_Mat (list): List of two global matrices, Glb_Mat[0] and Glb_Mat[1].
        alpha_frt (float): Scalar and a vector of parameters (batching).
        dt (float): Time step or similar scalar.
        poros_frt (float): Porosity for fractures.

    Returns:
        System_Mat (torch.sparse_coo_tensor): The assembled global system matrix (or a collection of matrices depend on the alpha_frt).
    """
    
    # Initialize Adiag
    Adiag = torch.zeros(node_frt, device=device)
    Adiag[0] = 2
    Adiag[-1] = 2
    Adiag[1:ny] = 4  # Assign 4 to indices 1 to ny-1

    # Initialize Alow and Aup
    Alow = torch.ones(node_frt, device=device)
    Aup = torch.ones(node_frt, device=device)

    # Create A_frt as a sparse tridiagonal matrix
    row = torch.cat([torch.arange(1, node_frt), torch.arange(node_frt), torch.arange(node_frt-1)])
    col = torch.cat([torch.arange(node_frt-1), torch.arange(node_frt), torch.arange(1, node_frt)])
    data = torch.cat([Alow[1:], Adiag, Aup[:-1]])
    A_frt = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),  # Indices need to be stacked
            values=(h / 6) * data,
            size=(node_frt, node_frt),
            dtype=torch.float64,  # Set appropriate data type
            device=device
        )

    # Create B_frt as a sparse matrix with diagonals at -1 and 0
    Bdiag = torch.ones(ny, device=device)  # Main diagonal
    Blow = -1*torch.ones(ny, device=device)  # Lower diagonal (only ny-1 elements)

    # Create row and column indices for the sparse matrix
    row_indices = torch.cat([torch.arange(ny), torch.arange(1, ny+1)])  # Main diagonal and lower diagonal
    col_indices = torch.cat([torch.arange(ny), torch.arange(ny)])   # Column indices for main and lower diagonal

    # Combine row and column indices into a single tensor
    indices = torch.stack([row_indices, col_indices])

    # Combine data for the sparse tensor
    data = torch.cat([Bdiag, Blow])  # Main diagonal and lower diagonal values

    # Create the sparse COO tensor
    B_frt = torch.sparse_coo_tensor(indices, data, size=(node_frt, ny), dtype=torch.float64, device=device)
    # Create the sparse COO tensor
    # Create C_frt as a diagonal sparse matrix
    C_frt = poros_frt*h*torch.sparse_coo_tensor(torch.stack([torch.arange(ny), torch.arange(ny)]),\
                                                torch.ones(ny), size=(ny, ny), dtype=torch.float64, device=device)

    # Create Glb_Mat_frt as a block matrix
#     A_frt_inv = (alpha_frt)**-1 * A_frt
    
    
#     # Use torch.block_diag on dense tensors
    num_alph = alpha_frt.shape[0]
    A_frt_inv = (alpha_frt[:, None, None]) ** -1 * A_frt  # Shape: [M, N, N]

    # Expand B_frt and C_frt to include batch dimensions
#     B_frt_expanded = B_frt.unsqueeze(0).expand(A_frt_inv.shape[0], -1, -1)  # Shape: [M, N, P]
#     C_frt_expanded = C_frt.unsqueeze(0).expand(A_frt_inv.shape[0], -1, -1)  # Shape: [M, P, P]
    B_frt_batched = [B_frt for _ in range(A_frt_inv.shape[0])]
    C_frt_batched = [C_frt for _ in range(A_frt_inv.shape[0])]
    
    # Convert the list into a batched sparse tensor
    B_frt_batched = torch.stack(B_frt_batched)
    C_frt_batched = torch.stack(C_frt_batched)
    # First row blocks: [A_frt_inv, B_frt_expanded]
    row1 = torch.cat((A_frt_inv, B_frt_batched), dim=2)  # Shape: [M, N, N + P]

    # Second row blocks: [dt * B_frt_expanded.transpose(1, 2), -C_frt_expanded]
    row2 = torch.cat((dt * B_frt_batched.permute(0, 2, 1), -C_frt_batched), dim=2)  # Shape: [M, P, N + P]

    # Combine both rows into a block matrix
    Glb_Mat_frt = torch.cat((row1, row2), dim=1)  # Shape: [M, N + P, N + P]

    # Convert to dense if necessary (already dense in this setup)
    Glb_Mat_frt = Glb_Mat_frt.to_dense()
    
    total_size = TriMesh[0].DOFs + TriMesh[1].DOFs + 2 * ny + 1
    
#     print(num_alph)
    System_Mat = torch.zeros((num_alph, total_size, total_size), dtype=torch.float64, device=device)  # PyTorch has no LIL matrix, so use dense

    # Assign Glb_Mat to blocks
    block1_start = 0
    block1_end = TriMesh[0].DOFs
    block2_start = block1_end
    block2_end = block2_start + TriMesh[1].DOFs
    block_frt_start = block2_end
    block_frt_end = block_frt_start + (2 * ny + 1)

    # Use slicing for assignments
    System_Mat[:, block1_start:block1_end, block1_start:block1_end] = Glb_Mat1st
    System_Mat[:, block2_start:block2_end, block2_start:block2_end] = Glb_Mat2nd
    System_Mat[:, block_frt_start:block_frt_end, block_frt_start:block_frt_end] = Glb_Mat_frt
    # Compute number of fracture edges for each mesh
    num_frt_egs = [TriMesh[k].FractureEgs.shape[1] for k in range(2)]

    # Initialize Com_matrix as a list of two sparse matrices
    Com_matrix = [torch.zeros((TriMesh[k].NumEgs, ny), dtype=torch.float64, device=device) for k in range(2)]

    # Assemble Com_matrix
    for k in range(2):
        frt_egs = TriMesh[k].FractureEgs.flatten()
        num_frt = num_frt_egs[k]

        if num_frt > ny:
            raise ValueError(f"Number of fracture edges {num_frt} exceeds ny={ny}.")

        # Assign h to (frt_egs[i], i) for i in range(num_frt)
        rows = frt_egs
        cols = torch.arange(num_frt)
        data = h * torch.ones(num_frt, dtype=torch.float64, device=device)
        
        for i in range(len(rows)):
            Com_matrix[k][rows[i], cols[i]] += data[i]
        # Convert to CSR-like for efficient ops
        Com_matrix[k] = Com_matrix[k].to_sparse()

    # Assign Com_matrix to System_Mat
    row_start_1 = 0
    row_end_1 = TriMesh[0].NumEgs
    col_start_1 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_1 = col_start_1 + ny
    System_Mat[:, row_start_1:row_end_1, col_start_1:col_end_1] = Com_matrix[0].to_dense()
    System_Mat[:, col_start_1:col_end_1, row_start_1:row_end_1] = dt * Com_matrix[0].transpose(0, 1).to_dense()

    row_start_2 = TriMesh[0].DOFs
    row_end_2 = TriMesh[0].DOFs + TriMesh[1].NumEgs
    col_start_2 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_2 = col_start_2 + ny
    
#     print([row_start_2, row_end_2, col_start_2, col_end_2])
    System_Mat[:, row_start_2:row_end_2, col_start_2:col_end_2] = Com_matrix[1].to_dense()
    System_Mat[:, col_start_2:col_end_2, row_start_2:row_end_2] = dt*Com_matrix[1].transpose(0, 1).to_dense()
#     print(System_Mat[col_start_2:col_end_2, row_start_2:row_end_2].to_sparse())
    # Convert System_Mat to a sparse tensor
    System_Mat = System_Mat.to_sparse()
#     System_Mat = System_Mat.to(torch.float64)
    return System_Mat

def Assemble_StiffMax_PureDiff_Alpha_frt(TriMesh, h, node_frt, ny, Glb_Mat, alpha_frt, dt, poros_frt, device):
    """
    Assemble the global system matrix for the Darcy equation with fractures using PyTorch.
    
    Parameters:
        TriMesh (list): List of two TriMesh objects.
        h (float): Spatial step or similar scalar.
        node_frt (int): Number of fracture nodes.
        ny (int): Number of y-direction nodes or similar.
        Glb_Mat (list): List of two global matrices, Glb_Mat[0] and Glb_Mat[1].
        alpha_frt (float): Scalar parameter.
        dt (float): Time step or similar scalar.
        poros_frt (float): Porosity for fractures.

    Returns:
        System_Mat (torch.sparse_coo_tensor): The assembled global system matrix.
    """
    
    # Initialize Adiag
    Adiag = torch.zeros(node_frt, device=device)
    Adiag[0] = 2
    Adiag[-1] = 2
    Adiag[1:ny] = 4  # Assign 4 to indices 1 to ny-1

    # Initialize Alow and Aup
    Alow = torch.ones(node_frt, device=device)
    Aup = torch.ones(node_frt, device=device)

    # Create A_frt as a sparse tridiagonal matrix
    row = torch.cat([torch.arange(1, node_frt), torch.arange(node_frt), torch.arange(node_frt-1)])
    col = torch.cat([torch.arange(node_frt-1), torch.arange(node_frt), torch.arange(1, node_frt)])
    data = torch.cat([Alow[1:], Adiag, Aup[:-1]])
    A_frt = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),  # Indices need to be stacked
            values=(h / 6) * data,
            size=(node_frt, node_frt),
            dtype=torch.float64,  # Set appropriate data type
            device=device
        )

    # Create B_frt as a sparse matrix with diagonals at -1 and 0
    Bdiag = torch.ones(ny, device=device)  # Main diagonal
    Blow = -1*torch.ones(ny, device=device)  # Lower diagonal (only ny-1 elements)

    # Create row and column indices for the sparse matrix
    row_indices = torch.cat([torch.arange(ny), torch.arange(1, ny+1)])  # Main diagonal and lower diagonal
    col_indices = torch.cat([torch.arange(ny), torch.arange(ny)])   # Column indices for main and lower diagonal

    # Combine row and column indices into a single tensor
    indices = torch.stack([row_indices, col_indices])

    # Combine data for the sparse tensor
    data = torch.cat([Bdiag, Blow])  # Main diagonal and lower diagonal values

    # Create the sparse COO tensor
    B_frt = torch.sparse_coo_tensor(indices, data, size=(node_frt, ny), dtype=torch.float64, device=device)
    # Create the sparse COO tensor
    # Create C_frt as a diagonal sparse matrix
    C_frt = poros_frt*h*torch.sparse_coo_tensor(torch.stack([torch.arange(ny), torch.arange(ny)]),\
                                                torch.ones(ny), size=(ny, ny), dtype=torch.float64, device=device)

    # Create Glb_Mat_frt as a block matrix
    A_frt_inv = (alpha_frt)**-1 * A_frt
    
    
    # Use torch.block_diag on dense tensors
    row1 = torch.cat((A_frt_inv, B_frt), dim=1)

    # Second row: [dt * B_frt.transpose(), -C_frt]
    row2 = torch.cat((dt * B_frt.transpose(0, 1), -C_frt), dim=1)

    # Combine both rows into a block matrix
    Glb_Mat_frt = torch.cat((row1, row2), dim=0)
    Glb_Mat_frt = Glb_Mat_frt.to_dense()

    total_size = TriMesh[0].DOFs + TriMesh[1].DOFs + 2 * ny + 1
    System_Mat = torch.zeros((total_size, total_size), dtype=torch.float64, device=device)  # PyTorch has no LIL matrix, so use dense

    # Assign Glb_Mat to blocks
#     System_Mat[0:TriMesh[0].DOFs, 0:TriMesh[1].DOFs] = Glb_Mat[0]
#     System_Mat[TriMesh[0].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs,
#                TriMesh[0].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs] = Glb_Mat[1]
#     System_Mat[TriMesh[0].DOFs + TriMesh[1].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs + (2 * ny + 1),
#                TriMesh[0].DOFs + TriMesh[1].DOFs:TriMesh[0].DOFs + TriMesh[1].DOFs + (2 * ny + 1)] = Glb_Mat_frt
    block1_start = 0
    block1_end = TriMesh[0].DOFs
    block2_start = block1_end
    block2_end = block2_start + TriMesh[1].DOFs
    block_frt_start = block2_end
    block_frt_end = block_frt_start + (2 * ny + 1)

    # Use slicing for assignments
    System_Mat[block1_start:block1_end, block1_start:block1_end] = Glb_Mat[0]
    System_Mat[block2_start:block2_end, block2_start:block2_end] = Glb_Mat[1]
    System_Mat[block_frt_start:block_frt_end, block_frt_start:block_frt_end] = Glb_Mat_frt
    # Compute number of fracture edges for each mesh
    num_frt_egs = [TriMesh[k].FractureEgs.shape[1] for k in range(2)]

    # Initialize Com_matrix as a list of two sparse matrices
    Com_matrix = [torch.zeros((TriMesh[k].NumEgs, ny), dtype=torch.float64, device=device) for k in range(2)]

    # Assemble Com_matrix
    for k in range(2):
        frt_egs = TriMesh[k].FractureEgs.flatten()
        num_frt = num_frt_egs[k]

        if num_frt > ny:
            raise ValueError(f"Number of fracture edges {num_frt} exceeds ny={ny}.")

        # Assign h to (frt_egs[i], i) for i in range(num_frt)
        rows = frt_egs
        cols = torch.arange(num_frt)
        data = h * torch.ones(num_frt, dtype=torch.float64, device=device)
        
        for i in range(len(rows)):
            Com_matrix[k][rows[i], cols[i]] += data[i]
        # Convert to CSR-like for efficient ops
        Com_matrix[k] = Com_matrix[k].to_sparse()

    # Assign Com_matrix to System_Mat
    row_start_1 = 0
    row_end_1 = TriMesh[0].NumEgs
    col_start_1 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_1 = col_start_1 + ny
    System_Mat[row_start_1:row_end_1, col_start_1:col_end_1] = Com_matrix[0].to_dense()
    System_Mat[col_start_1:col_end_1, row_start_1:row_end_1] = dt * Com_matrix[0].transpose(0, 1).to_dense()

    row_start_2 = TriMesh[0].DOFs
    row_end_2 = TriMesh[0].DOFs + TriMesh[1].NumEgs
    col_start_2 = TriMesh[0].DOFs + TriMesh[1].DOFs + node_frt
    col_end_2 = col_start_2 + ny
    
#     print([row_start_2, row_end_2, col_start_2, col_end_2])
    System_Mat[row_start_2:row_end_2, col_start_2:col_end_2] = Com_matrix[1].to_dense()
    System_Mat[col_start_2:col_end_2, row_start_2:row_end_2] = dt*Com_matrix[1].transpose(0, 1).to_dense()
#     print(System_Mat[col_start_2:col_end_2, row_start_2:row_end_2].to_sparse())
    # Convert System_Mat to a sparse tensor
    System_Mat = System_Mat.to_sparse()
#     System_Mat = System_Mat.to(torch.float64)
    return System_Mat


def Assemble_RHS_PureDiff_DA_General(
    p0, p0_frt, ensemble_size, DOFs, ny, DirichletEgs, NeumannEgs, FractureEgs, EqnBC_Subdom, EqnBC_fracture,
    TriMesh, GAUSSQUAD, System_Mat, dt, pres_bot_frt, pres_top_frt, presBC_subdom, poros_frt, poros_subdom, noise, device
):
    """
    Assemble the global right-hand side (RHS) vector for a pure diffusion test case using PyTorch.
    """

    # Initialize mesh information lists
    NumEms = [0, 0]
    NumEgs = [0, 0]
    area = [None, None]
    LenEg = [None, None]
    sn = [None, None]

    NumFractureEgs = [0, 0]
    NumNeumannEgs = [0, 0]
    NumDirichletEgs = [0, 0]

    # Extract mesh information for each subdomain
    for k in range(2):
        NumEms[k] = TriMesh[k].NumEms
        NumEgs[k] = TriMesh[k].NumEgs
        area[k] = TriMesh[k].area
        LenEg[k] = TriMesh[k].LenEg
        sn[k] = TriMesh[k].SignEmEg

    # Count boundary edges for each subdomain
    for k in range(2):
        if DirichletEgs[k].dim() == 1:  # If it's 1D (shape is [1])
            NumDirichletEgs[k] = DirichletEgs[k].shape[0]  # Use shape[0]
        else:  # If it's 2D (shape is [1, N])
            NumDirichletEgs[k] = DirichletEgs[k].shape[1]  # Use shape[1]
            
    for k in range(2):
        NumFractureEgs[k] = FractureEgs[k].shape[1]
        NumNeumannEgs[k] = NeumannEgs[k].shape[1]

    # Initialize the global RHS vector using PyTorch
    total_size = DOFs[0] + DOFs[1] + 2 * ny + 1
    GlbRHS = torch.zeros((total_size, ensemble_size), dtype=torch.float64, device=device)

    # Initialize local RHS vectors and fracture RHS
    Local_RHS = [torch.zeros((DOFs[0], ensemble_size), dtype=torch.float64, device=device), \
                 torch.zeros((DOFs[1], ensemble_size), dtype=torch.float64, device=device)]
    Fracture_RHS = torch.zeros((ny, ensemble_size), dtype=torch.float64, device=device)
    NumQuadPts_TRIG = GAUSSQUAD['TRIG'].shape[0]
         
    # Assemble local RHS for each subdomain
    for o in range(2):
        for k in range(NumQuadPts_TRIG):
            qp = (
                GAUSSQUAD['TRIG'][k, 0] * TriMesh[o].node[TriMesh[o].elem[:, 0], :] +
                GAUSSQUAD['TRIG'][k, 1] * TriMesh[o].node[TriMesh[o].elem[:, 1], :] +
                GAUSSQUAD['TRIG'][k, 2] * TriMesh[o].node[TriMesh[o].elem[:, 2], :]
            )
            RHS = GAUSSQUAD['TRIG'][k, 3] * EqnBC_Subdom[o].fxnf(qp, 0) * area[o]
            Local_RHS[o][NumEgs[o]:(NumEgs[o] + NumEms[o]), :] -= RHS.unsqueeze(1)

    # Assign contributions to GlbRHS for each subdomain
    RHS_1subdom = dt*Local_RHS[0][NumEgs[0]:(NumEgs[0] + NumEms[0]), :]-poros_subdom*p0[0]*area[0].unsqueeze(1)
    RHS_2subdom = dt*Local_RHS[1][NumEgs[1]:(NumEgs[1] + NumEms[1]), :]-poros_subdom*p0[1]*area[1].unsqueeze(1)
    
    if noise.dim() ==1:
        noise = noise.unsqueeze(1)
        
    GlbRHS[NumEgs[0]:(NumEgs[0] + NumEms[0]), :] += RHS_1subdom + noise[0:NumEms[0], :]
    GlbRHS[(DOFs[0] + NumEgs[1]):(DOFs[0] + (NumEgs[1] + NumEms[1])), :] += RHS_2subdom+noise[NumEms[0]:(NumEms[0]+NumEms[1]), :]

    # Handle fracture contributions
    y1 = TriMesh[0].node[TriMesh[0].edge[FractureEgs[0], 0], 1]
    y2 = TriMesh[0].node[TriMesh[0].edge[FractureEgs[0], 1], 1]
    LengthEgs = torch.abs(y2 - y1)
    y1 = y1.to(device)
    y2 = y2.to(device)
    LengthEgs = LengthEgs.to(device)
    qp = torch.zeros(NumFractureEgs[0], dtype=torch.float64, device=device)
    NumQuadPts_LINE = GAUSSQUAD['LINE'].shape[0]

    for k in range(NumQuadPts_LINE):
        qp = GAUSSQUAD['LINE'][k, 0] * y1 + GAUSSQUAD['LINE'][k, 1] * y2
        Fracture_RHS -= GAUSSQUAD['LINE'][k, 2]*EqnBC_fracture.fxnf(qp.reshape(-1, 1), 0)*LengthEgs.reshape(-1, 1)

#     GlbRHS[DOFs[0] + DOFs[1], :] += pres_bot_frt
#     GlbRHS[DOFs[0] + DOFs[1] + ny, :] -= pres_top_frt
    RHS_frt = dt * Fracture_RHS - poros_frt*p0_frt*LengthEgs.reshape(-1, 1)
    GlbRHS[(DOFs[0] + DOFs[1] + ny + 1):(DOFs[0] + DOFs[1] + 2 * ny + 1), :] +=\
                                                RHS_frt+noise[(NumEms[0]+NumEms[1]):(NumEms[0]+NumEms[1]+ny), :]

    # Boundary RHS (Dirichlet)
    boundary_RHS = [torch.zeros((NumDirichletEgs[k], ensemble_size), dtype=torch.float64, device=device) for k in range(2)]

    for k in range(2):
        u1 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 0], 0]
        v1 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 0], 1]
        u2 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 1], 0]
        v2 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 1], 1]
        u1 = u1.to(device)
        v1 = v1.to(device)
        u2 = u2.to(device)
        v2 = v2.to(device)
        LenDirichletEg = torch.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
        qp = torch.zeros((NumDirichletEgs[k], 2), dtype=torch.float64, device=device)
        for o in range(NumQuadPts_LINE):
#             print(GAUSSQUAD['LINE'][o, 0]*u1.reshape(-1, 1))
            qp[:, 0] = GAUSSQUAD['LINE'][o, 0]*u1+GAUSSQUAD['LINE'][o, 1]*u2
            qp[:, 1] = GAUSSQUAD['LINE'][o, 0]*v1+GAUSSQUAD['LINE'][o, 1]*v2
            BCs = GAUSSQUAD['LINE'][o, 2]*presBC_subdom[k]*LenDirichletEg.reshape(-1, 1)
            boundary_RHS[k] -= BCs

    GlbRHS[DirichletEgs[0], :] += boundary_RHS[0]
    GlbRHS[DOFs[0] + DirichletEgs[1], :] += boundary_RHS[1]
    
    # Neumann Boundary
    flag = torch.zeros((DOFs[0] + DOFs[1] + 2 * ny + 1, 1), dtype=torch.float64, device=device)
    flag[NeumannEgs[0]] += 1
    flag[DOFs[0] + NeumannEgs[1]] += 1
    flag[DOFs[0]+DOFs[1]] += 1
    flag[DOFs[0]+DOFs[1]+ny] += 1
    
    sln = torch.zeros((DOFs[0] + DOFs[1] + 2 * ny + 1, ensemble_size), dtype=torch.float64, device=device)
    sln[NeumannEgs[0], :] += torch.zeros((NumNeumannEgs[0], ensemble_size), dtype=torch.float64, device=device)
    sln[DOFs[0] + NeumannEgs[1], :] += torch.zeros((NumNeumannEgs[1], ensemble_size), dtype=torch.float64, device=device)
    sln[DOFs[0]+DOFs[1], :] = 0;
    sln[DOFs[0]+DOFs[1]+ny, :] = 0;
    # Reducing RHS
    
    GlbRHS -= torch.matmul(System_Mat, sln)

    return GlbRHS, flag, sln

def Assemble_RHS_PureDiff_DA_General_Aniso(
    p0, p0_frt, ensemble_size, DOFs, ny, DirichletEgs, NeumannEgs, FractureEgs, EqnBC_Subdom, EqnBC_fracture,
    TriMesh, GAUSSQUAD, System_Mat, dt, pres_bot_frt, pres_top_frt, presBC_subdom, poros_frt, poros_subdom, noise, device
):
    """
    Assemble the global right-hand side (RHS) vector for a pure diffusion test case using PyTorch.
    """

    # Initialize mesh information lists
    NumEms = [0, 0]
    NumEgs = [0, 0]
    area = [None, None]
    LenEg = [None, None]
    sn = [None, None]

    NumFractureEgs = [0, 0]
    NumNeumannEgs = [0, 0]
    NumDirichletEgs = [0, 0]

    # Extract mesh information for each subdomain
    for k in range(2):
        NumEms[k] = TriMesh[k].NumEms
        NumEgs[k] = TriMesh[k].NumEgs
        area[k] = TriMesh[k].area
        LenEg[k] = TriMesh[k].LenEg
        sn[k] = TriMesh[k].SignEmEg

    # Count boundary edges for each subdomain
    for k in range(2):
        if DirichletEgs[k].dim() == 1:  # If it's 1D (shape is [1])
            NumDirichletEgs[k] = DirichletEgs[k].shape[0]  # Use shape[0]
        else:  # If it's 2D (shape is [1, N])
            NumDirichletEgs[k] = DirichletEgs[k].shape[1]  # Use shape[1]
            
    for k in range(2):
        NumFractureEgs[k] = FractureEgs[k].shape[1]
        NumNeumannEgs[k] = NeumannEgs[k].shape[1]

    # Initialize the global RHS vector using PyTorch
    total_size = DOFs[0] + DOFs[1] + 2 * ny + 1
    GlbRHS = torch.zeros((total_size, ensemble_size), dtype=torch.float64, device=device)

    # Initialize local RHS vectors and fracture RHS
    Local_RHS = [torch.zeros((DOFs[0], ensemble_size), dtype=torch.float64, device=device), \
                 torch.zeros((DOFs[1], ensemble_size), dtype=torch.float64, device=device)]
    Fracture_RHS = torch.zeros((ny, ensemble_size), dtype=torch.float64, device=device)
    NumQuadPts_TRIG = GAUSSQUAD['TRIG'].shape[0]
         
    # Assemble local RHS for each subdomain
    for o in range(2):
        for k in range(NumQuadPts_TRIG):
            qp = (
                GAUSSQUAD['TRIG'][k, 0] * TriMesh[o].node[TriMesh[o].elem[:, 0], :] +
                GAUSSQUAD['TRIG'][k, 1] * TriMesh[o].node[TriMesh[o].elem[:, 1], :] +
                GAUSSQUAD['TRIG'][k, 2] * TriMesh[o].node[TriMesh[o].elem[:, 2], :]
            )
            RHS = GAUSSQUAD['TRIG'][k, 3] * EqnBC_Subdom[o].fxnf(qp, 0) * area[o]
            Local_RHS[o][NumEgs[o]:(NumEgs[o] + NumEms[o]), :] -= RHS.unsqueeze(1)

    # Assign contributions to GlbRHS for each subdomain
    RHS_1subdom = dt*Local_RHS[0][NumEgs[0]:(NumEgs[0] + NumEms[0]), :]-poros_subdom*p0[0]*area[0].unsqueeze(1)
    RHS_2subdom = dt*Local_RHS[1][NumEgs[1]:(NumEgs[1] + NumEms[1]), :]-poros_subdom*p0[1]*area[1].unsqueeze(1)
    
    if noise.dim() ==1:
        noise = noise.unsqueeze(1)
        
    GlbRHS[NumEgs[0]:(NumEgs[0] + NumEms[0]), :] += RHS_1subdom + noise[0:NumEms[0], :]
    GlbRHS[(DOFs[0] + NumEgs[1]):(DOFs[0] + (NumEgs[1] + NumEms[1])), :] += RHS_2subdom+noise[NumEms[0]:(NumEms[0]+NumEms[1]), :]

    # Handle fracture contributions
    y1 = TriMesh[0].node[TriMesh[0].edge[FractureEgs[0], 0], 1]
    y2 = TriMesh[0].node[TriMesh[0].edge[FractureEgs[0], 1], 1]
    LengthEgs = torch.abs(y2 - y1)
    y1 = y1.to(device)
    y2 = y2.to(device)
    LengthEgs = LengthEgs.to(device)
    qp = torch.zeros(NumFractureEgs[0], dtype=torch.float64, device=device)
    NumQuadPts_LINE = GAUSSQUAD['LINE'].shape[0]

    for k in range(NumQuadPts_LINE):
        qp = GAUSSQUAD['LINE'][k, 0] * y1 + GAUSSQUAD['LINE'][k, 1] * y2
        Fracture_RHS -= GAUSSQUAD['LINE'][k, 2]*EqnBC_fracture.fxnf(qp.reshape(-1, 1), 0)*LengthEgs.reshape(-1, 1)

    GlbRHS[DOFs[0] + DOFs[1], :] += pres_bot_frt
    GlbRHS[DOFs[0] + DOFs[1] + ny, :] -= pres_top_frt
    RHS_frt = dt * Fracture_RHS - poros_frt*p0_frt*LengthEgs.reshape(-1, 1)
    GlbRHS[(DOFs[0] + DOFs[1] + ny + 1):(DOFs[0] + DOFs[1] + 2 * ny + 1), :] +=\
                                                RHS_frt+noise[(NumEms[0]+NumEms[1]):(NumEms[0]+NumEms[1]+ny), :]

    # Boundary RHS (Dirichlet)
    boundary_RHS = [torch.zeros((NumDirichletEgs[k], ensemble_size), dtype=torch.float64, device=device) for k in range(2)]

    for k in range(2):
        u1 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 0], 0]
        v1 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 0], 1]
        u2 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 1], 0]
        v2 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 1], 1]
        u1 = u1.to(device)
        v1 = v1.to(device)
        u2 = u2.to(device)
        v2 = v2.to(device)
        LenDirichletEg = torch.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
        qp = torch.zeros((NumDirichletEgs[k], 2), dtype=torch.float64, device=device)
        for o in range(NumQuadPts_LINE):
#             print(GAUSSQUAD['LINE'][o, 0]*u1.reshape(-1, 1))
            qp[:, 0] = GAUSSQUAD['LINE'][o, 0]*u1+GAUSSQUAD['LINE'][o, 1]*u2
            qp[:, 1] = GAUSSQUAD['LINE'][o, 0]*v1+GAUSSQUAD['LINE'][o, 1]*v2
            BCs = GAUSSQUAD['LINE'][o, 2]*presBC_subdom[k]*LenDirichletEg.reshape(-1, 1)
            boundary_RHS[k] -= BCs

    GlbRHS[DirichletEgs[0], :] += boundary_RHS[0]
    GlbRHS[DOFs[0] + DirichletEgs[1], :] += boundary_RHS[1]
    
    # Neumann Boundary
    flag = torch.zeros((DOFs[0] + DOFs[1] + 2 * ny + 1, 1), dtype=torch.float64, device=device)
    flag[NeumannEgs[0]] += 1
    flag[DOFs[0] + NeumannEgs[1]] += 1
    
    sln = torch.zeros((DOFs[0] + DOFs[1] + 2 * ny + 1, ensemble_size), dtype=torch.float64, device=device)
    sln[NeumannEgs[0], :] += torch.zeros((NumNeumannEgs[0], ensemble_size), dtype=torch.float64, device=device)
    sln[DOFs[0] + NeumannEgs[1], :] += torch.zeros((NumNeumannEgs[1], ensemble_size), dtype=torch.float64, device=device)
    # Reducing RHS
    
    GlbRHS -= torch.matmul(System_Mat, sln)

    return GlbRHS, flag, sln
    
def Assemble_RHS_PureDiff_DA_Vectorized(
    p0, p0_frt, ensemble_size, DOFs, ny, DirichletEgs, NeumannEgs, FractureEgs, EqnBC_Subdom, EqnBC_fracture,
    TriMesh, GAUSSQUAD, System_Mat, dt, pres_bot_frt, pres_top_frt, presBC_subdom, poros_frt, poros_subdom, noise, device
):
    """
    Assemble the global right-hand side (RHS) vector for a pure diffusion test case using PyTorch.
    """
    num_batch = System_Mat.shape[0]
    # Initialize mesh information lists
    NumEms = [0, 0]
    NumEgs = [0, 0]
    area = [None, None]
    LenEg = [None, None]
    sn = [None, None]

    NumFractureEgs = [0, 0]
    NumNeumannEgs = [0, 0]
    NumDirichletEgs = [0, 0]

    # Extract mesh information for each subdomain
    for k in range(2):
        NumEms[k] = TriMesh[k].NumEms
        NumEgs[k] = TriMesh[k].NumEgs
        area[k] = TriMesh[k].area
        LenEg[k] = TriMesh[k].LenEg
        sn[k] = TriMesh[k].SignEmEg

    # Count boundary edges for each subdomain
    for k in range(2):
        if DirichletEgs[k].dim() == 1:  # If it's 1D (shape is [1])
            NumDirichletEgs[k] = DirichletEgs[k].shape[0]  # Use shape[0]
        else:  # If it's 2D (shape is [1, N])
            NumDirichletEgs[k] = DirichletEgs[k].shape[1]  # Use shape[1]
            
    for k in range(2):
#         NumDirichletEgs[k] = DirichletEgs[k].shape[1]
        NumFractureEgs[k] = FractureEgs[k].shape[1]
        NumNeumannEgs[k] = NeumannEgs[k].shape[1]

    # Initialize the global RHS vector using PyTorch
    total_size = DOFs[0] + DOFs[1] + 2 * ny + 1
    GlbRHS = torch.zeros((num_batch, total_size, ensemble_size), dtype=torch.float64, device=device)

    # Initialize local RHS vectors and fracture RHS
    Local_RHS = [torch.zeros((DOFs[0], ensemble_size), dtype=torch.float64, device=device), \
                 torch.zeros((DOFs[1], ensemble_size), dtype=torch.float64, device=device)]
    Fracture_RHS = torch.zeros((ny, ensemble_size), dtype=torch.float64, device=device)
    NumQuadPts_TRIG = GAUSSQUAD['TRIG'].shape[0]
         
    # Assemble local RHS for each subdomain
    for o in range(2):
        for k in range(NumQuadPts_TRIG):
            qp = (
                GAUSSQUAD['TRIG'][k, 0] * TriMesh[o].node[TriMesh[o].elem[:, 0], :] +
                GAUSSQUAD['TRIG'][k, 1] * TriMesh[o].node[TriMesh[o].elem[:, 1], :] +
                GAUSSQUAD['TRIG'][k, 2] * TriMesh[o].node[TriMesh[o].elem[:, 2], :]
            )
#             qp = qp.to(device)
            RHS = GAUSSQUAD['TRIG'][k, 3] * EqnBC_Subdom[o].fxnf(qp, 0) * area[o]
#             RHS = RHS.to(device)
#             RHS = RHS.to(device)
#             print(RHS.unsqueeze(1).shape)
            Local_RHS[o][NumEgs[o]:(NumEgs[o] + NumEms[o]), :] -= RHS.unsqueeze(1)

    # Assign contributions to GlbRHS for each subdomain
#     print(p0[0].shape)
#     print(area[0].unsqueeze(1).shape)
    RHS_1subdom = dt*Local_RHS[0][NumEgs[0]:(NumEgs[0] + NumEms[0]), :]-poros_subdom*p0[0]*area[0].unsqueeze(1)
    RHS_2subdom = dt*Local_RHS[1][NumEgs[1]:(NumEgs[1] + NumEms[1]), :]-poros_subdom*p0[1]*area[1].unsqueeze(1)
    
    if noise.dim() ==1:
        noise = noise.unsqueeze(1)
        
    GlbRHS[:, NumEgs[0]:(NumEgs[0] + NumEms[0]), :] += RHS_1subdom + noise[0:NumEms[0], :]
    GlbRHS[:, (DOFs[0] + NumEgs[1]):(DOFs[0] + (NumEgs[1] + NumEms[1])), :] += RHS_2subdom+noise[NumEms[0]:(NumEms[0]+NumEms[1]), :]

    # Handle fracture contributions
    y1 = TriMesh[0].node[TriMesh[0].edge[FractureEgs[0], 0], 1]
    y2 = TriMesh[0].node[TriMesh[0].edge[FractureEgs[0], 1], 1]
    LengthEgs = torch.abs(y2 - y1)
    y1 = y1.to(device)
    y2 = y2.to(device)
    LengthEgs = LengthEgs.to(device)
    qp = torch.zeros(NumFractureEgs[0], dtype=torch.float64, device=device)
    NumQuadPts_LINE = GAUSSQUAD['LINE'].shape[0]
#     print(Fracture_RHS.shape)
#     print(LengthEgs.shape)
    for k in range(NumQuadPts_LINE):
        qp = GAUSSQUAD['LINE'][k, 0] * y1 + GAUSSQUAD['LINE'][k, 1] * y2
#         print(qp)
#         print(EqnBC_fracture.fxnf(qp, 0).shape)
        Fracture_RHS -= GAUSSQUAD['LINE'][k, 2]*EqnBC_fracture.fxnf(qp.reshape(-1, 1), 0)*LengthEgs.reshape(-1, 1)

    GlbRHS[:, DOFs[0] + DOFs[1], :] += pres_bot_frt
    GlbRHS[:, DOFs[0] + DOFs[1] + ny, :] -= pres_top_frt
    RHS_frt = dt * Fracture_RHS - poros_frt*p0_frt*LengthEgs.reshape(-1, 1)
    GlbRHS[:, (DOFs[0] + DOFs[1] + ny + 1):(DOFs[0] + DOFs[1] + 2 * ny + 1), :] +=\
                                                RHS_frt+noise[(NumEms[0]+NumEms[1]):(NumEms[0]+NumEms[1]+ny), :]

    # Boundary RHS (Dirichlet)
    boundary_RHS = [torch.zeros((NumDirichletEgs[k], ensemble_size), dtype=torch.float64, device=device) for k in range(2)]

    for k in range(2):
        u1 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 0], 0]
        v1 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 0], 1]
        u2 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 1], 0]
        v2 = TriMesh[k].node[TriMesh[k].edge[DirichletEgs[k].flatten(), 1], 1]
        u1 = u1.to(device)
        v1 = v1.to(device)
        u2 = u2.to(device)
        v2 = v2.to(device)
        LenDirichletEg = torch.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
        qp = torch.zeros((NumDirichletEgs[k], 2), dtype=torch.float64, device=device)
        for o in range(NumQuadPts_LINE):
#             print(GAUSSQUAD['LINE'][o, 0]*u1.reshape(-1, 1))
            qp[:, 0] = GAUSSQUAD['LINE'][o, 0]*u1+GAUSSQUAD['LINE'][o, 1]*u2
            qp[:, 1] = GAUSSQUAD['LINE'][o, 0]*v1+GAUSSQUAD['LINE'][o, 1]*v2
            BCs = GAUSSQUAD['LINE'][o, 2]*presBC_subdom[k]*LenDirichletEg.reshape(-1, 1)
            boundary_RHS[k] -= BCs

    GlbRHS[:, DirichletEgs[0], :] += boundary_RHS[0]
    GlbRHS[:, DOFs[0] + DirichletEgs[1], :] += boundary_RHS[1]
    
#     GlbRHS += noise
    # Neumann Boundary
    flag = torch.zeros((DOFs[0] + DOFs[1] + 2 * ny + 1, 1), dtype=torch.float64, device=device)
    flag[NeumannEgs[0]] += 1
    flag[DOFs[0] + NeumannEgs[1]] += 1

    sln = torch.zeros((DOFs[0] + DOFs[1] + 2 * ny + 1, ensemble_size), dtype=torch.float64, device=device)
    sln[NeumannEgs[0]] += torch.zeros((NumNeumannEgs[0], ensemble_size), dtype=torch.float64, device=device)
    sln[DOFs[0] + NeumannEgs[1]] += torch.zeros((NumNeumannEgs[1], ensemble_size), dtype=torch.float64, device=device)

    # Reducing RHS
    
    GlbRHS -= torch.matmul(System_Mat, sln)

    return GlbRHS, flag, sln

def RearrangeSol(sln, TriMesh, ny, node_frt):
    ReSln = torch.zeros_like(sln)
    DOFs_Local = [None] * 2
    NumEgs = [None]*2
    NumEms = [None]*2
    for k in range(2):
        DOFs_Local[k] = TriMesh[k].DOFs
        NumEms[k] = TriMesh[k].NumEms
        NumEgs[k] = TriMesh[k].NumEgs
        
    ReSln[:, :NumEgs[0]] += sln[:, :NumEgs[0]]
                             
    ReSln[:, NumEgs[0]+torch.arange(0, NumEgs[1])] +=sln[:, DOFs_Local[0]+torch.arange(0, NumEgs[1])]
          
    ReSln[:, NumEgs[0]+NumEgs[1]+torch.arange(0, node_frt)] += sln[:, DOFs_Local[0]+DOFs_Local[1]+torch.arange(0, node_frt)]
                             
    ReSln[:, NumEgs[0]+NumEgs[1]+node_frt+torch.arange(0, NumEms[0])] += sln[:, NumEgs[0]+torch.arange(0, NumEms[0])]
    
    ReSln[:, NumEgs[0]+NumEgs[1]+node_frt+NumEms[0]+torch.arange(0, NumEms[1])] +=\
                             sln[:, DOFs_Local[0]+NumEgs[1]+torch.arange(0, NumEms[1])]
    
    ReSln[:, NumEgs[0]+NumEgs[1]+node_frt+NumEms[0]+NumEms[1]+torch.arange(0, ny)] += \
                            sln[:, DOFs_Local[0]+DOFs_Local[1]+node_frt+torch.arange(0, ny)]
              
    return ReSln

def DecompSol(sln, TriMesh, ny, node_frt):
    timeshape = sln.shape[0]
    DOFs_Local = [None] * 2
    NumEgs = [None]*2
    NumEms = [None]*2
    for k in range(2):
        DOFs_Local[k] = TriMesh[k].DOFs
        NumEms[k] = TriMesh[k].NumEms
        NumEgs[k] = TriMesh[k].NumEgs
     
    Sln1 = torch.zeros(timeshape, DOFs_Local[0], dtype=torch.float64)
    Sln2 = torch.zeros(timeshape, DOFs_Local[1], dtype=torch.float64)
    Slnfrt = torch.zeros(timeshape, ny+node_frt, dtype=torch.float64)
    
    Sln1[:, :] += sln[:, :DOFs_Local[0]]
    
    Sln2[:, :] += sln[:, DOFs_Local[0]:DOFs_Local[0]+DOFs_Local[1]]
    
    Slnfrt[:, :] += sln[:, DOFs_Local[0]+DOFs_Local[1]:DOFs_Local[0]+DOFs_Local[1]+node_frt+ny]
              
    return Sln1, Sln2, Slnfrt

def ExtractPres(sln, TriMesh, ny, node_frt):
    TotalNumEms = TriMesh[0].NumEms+TriMesh[1].NumEms+ny
    ReSln = torch.zeros(TotalNumEms, dtype = torch.float64)
#     print(ReSln.shape)
#     print(sln.shape)
    DOFs_Local = [None] * 2
    NumEgs = [None]*2
    NumEms = [None]*2
    for k in range(2):
        DOFs_Local[k] = TriMesh[k].DOFs
        NumEms[k] = TriMesh[k].NumEms
        NumEgs[k] = TriMesh[k].NumEgs
                            
    ReSln[torch.arange(0, NumEms[0])] += sln[NumEgs[0]+torch.arange(0, NumEms[0])]
    
    ReSln[NumEms[0]+torch.arange(0, NumEms[1])] += sln[DOFs_Local[0]+NumEgs[1]+torch.arange(0, NumEms[1])]
    
    ReSln[NumEms[0]+NumEms[1]+torch.arange(0, ny)] += sln[DOFs_Local[0]+DOFs_Local[1]+node_frt+torch.arange(0, ny)]
              
    return ReSln

def RearrangeSol_Reverse(sln, TriMesh, ny, node_frt):
    ReSln = torch.zeros_like(sln)
    DOFs_Local = [None] * 2
    NumEgs = [None]*2
    NumEms = [None]*2
    for k in range(2):
        DOFs_Local[k] = TriMesh[k].DOFs
        NumEms[k] = TriMesh[k].NumEms
        NumEgs[k] = TriMesh[k].NumEgs
        
    ReSln[:, :NumEgs[0]] += sln[:, :NumEgs[0]]
                             
    ReSln[:, DOFs_Local[0]+torch.arange(0, NumEgs[1])] +=sln[:, NumEgs[0]+torch.arange(0, NumEgs[1])]
          
    ReSln[:, DOFs_Local[0]+DOFs_Local[1]+torch.arange(0, node_frt)] += sln[:, NumEgs[0]+NumEgs[1]+torch.arange(0, node_frt)]
                             
    ReSln[:, NumEgs[0]+torch.arange(0, NumEms[0])]+= sln[:, NumEgs[0]+NumEgs[1]+node_frt+torch.arange(0, NumEms[0])] 
    
    ReSln[:, DOFs_Local[0]+NumEgs[1]+torch.arange(0, NumEms[1])] +=\
                             sln[:, NumEgs[0]+NumEgs[1]+node_frt+NumEms[0]+torch.arange(0, NumEms[1])]
    
    ReSln[:, DOFs_Local[0]+DOFs_Local[1]+node_frt+torch.arange(0, ny)] += \
                            sln[:, NumEgs[0]+NumEgs[1]+node_frt+NumEms[0]+NumEms[1]+torch.arange(0, ny)]
              
    return ReSln

def ComputSol_DFFracture_Afrt_Ksubdom_DA_General(
    X0, ny, node_frt, TriMesh, nt, dt, poros_subdom, poros_frt, BCs_frt, BCs_Subdom,
    Glob_Mat, GAUSSQUAD, Eqn_Subdom, EqnBC_frt, SDE_sigma, device, noise, signal
):
    """
    Assemble and extract the global solution for a pure diffusion test case with fractures using PyTorch.

    Parameters:
        ny (int): Number of y-direction nodes or similar parameter.
        node_frt (int): Number of fracture nodes.
        TriMesh (list of objects): List of two TriMesh objects, each with required attributes.
        nt (int): Number of time steps.
        dt (float): Time step size.
        poros_subdom (float): Porosity for subdomains.
        poros_frt (float): Porosity for fractures.
        Glob_Mat (list of PyTorch sparse tensors): List of two global matrices, one per subdomain.
        GAUSSQUAD (dict): Dictionary containing quadrature points and weights.
        EqnBC_fracture (list of objects): List of two boundary condition objects for fractures.

    Returns:
        Glob_Sln (torch.Tensor): Assembled global solution matrix.
    """
    # Initialize lists for boundary edges and equation boundary conditions
    FractureEgs = [None] * 2
    DirichletEgs = [None] * 2
    NeumannEgs = [None] * 2
    EqnBC = [None] * 2
    
    for k in range(2):
        FractureEgs[k] = TriMesh[k].FractureEgs
        DirichletEgs[k] = TriMesh[k].DirichletEgs
        NeumannEgs[k] = TriMesh[k].NeumannEgs
        EqnBC[k]  = Eqn_Subdom[k]
        
    node = [None] * 2
    elem = [None] * 2
    edge = [None] * 2
    DOFs_Local = [None] * 2

    for k in range(2):
        node[k] = TriMesh[k].node
        elem[k] = TriMesh[k].elem
        edge[k] = TriMesh[k].edge
        DOFs_Local[k] = TriMesh[k].DOFs
    
#     print(DirichletEgs[0].shape)
    presBC_subdom = [torch.zeros(DirichletEgs[k].shape[1], 1, dtype=torch.float64, device=device) for k in range(2)]
    
    # Initial conditions
    m, ensemble_size = X0.shape
    X = torch.zeros_like(X0)
#     noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(X0)
    c0 = [None, None]
    
    c0[0] = X0[TriMesh[0].NumEgs + torch.arange(0, TriMesh[0].NumEms), :]
    c0[1] = X0[DOFs_Local[0] + TriMesh[1].NumEgs + torch.arange(0, TriMesh[1].NumEms), :]
    c0_frt = X0[DOFs_Local[0] + DOFs_Local[1] + node_frt + torch.arange(0, ny), :]
    
    C = torch.cat((c0[0], c0[1], c0_frt), dim=0)
    if signal == 0:
        noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(C)
          
    pres_top_frt = 0
    presBC_subdom[0][:, 0] = 0
    
#     for n in range(nt):  
    pres_bot_frt = BCs_frt
    presBC_subdom[1][:, 0] += BCs_Subdom

    # Solving
    GlbRHS, flag, sln = Assemble_RHS_PureDiff_DA_General(
        c0, c0_frt, ensemble_size, DOFs_Local, ny, DirichletEgs, NeumannEgs, FractureEgs,
        EqnBC, EqnBC_frt, TriMesh, GAUSSQUAD, Glob_Mat, dt, pres_bot_frt, pres_top_frt, 
        presBC_subdom, poros_frt, poros_subdom, noise, device
    )
    flag_flat = flag.view(-1).bool()
    EmFreeEg = torch.where(~flag_flat)[0]

    if Glob_Mat.device.type == 'cuda':
        Glob_Mat_cpu = (Glob_Mat.to_dense()).cpu()
    else:
        Glob_Mat_cpu = Glob_Mat.to_dense()

    Glob_Mat_scipy = scipy.sparse.csr_matrix(Glob_Mat_cpu.numpy())  # Or use sparse conversion

    if EmFreeEg.device.type == 'cuda':
        EmFreeEg_np = np.array(EmFreeEg.cpu().numpy())
    else:
        EmFreeEg_np = np.array(EmFreeEg.numpy())

    if GlbRHS.device.type =='cuda':
        GlbRHS_cpu = GlbRHS.cpu()
    else: 
        GlbRHS_cpu = GlbRHS

    # Solve using SciPy's sparse solver
    result = scipy.sparse.linalg.spsolve(Glob_Mat_scipy[np.ix_(EmFreeEg_np, EmFreeEg_np)], GlbRHS_cpu[EmFreeEg_np, :].numpy())

    # Convert result back to PyTorch tensor
    result = torch.from_numpy(result)
#         result = torch.linalg.solve(Glob_Mat[EmFreeEg][:, EmFreeEg], GlbRHS[EmFreeEg])
#         sln[EmFreeEg, :] = result.view(-1, 1)
#         sln[EmFreeEg, :] = result
    result = result.to(device)
    if result.dim() == 1:  # Check if result is 1D
        sln[EmFreeEg, :] += result.unsqueeze(1)  # Reshape to [4009, 1]
    else:  # result is already 2D
        sln[EmFreeEg, :] += result  # Assign directly
#         X += sln
    return sln

def ComputSol_DFFracture_Afrt_Ksubdom_DA_General_Aniso(
    X0, ny, node_frt, TriMesh, nt, dt, poros_subdom, poros_frt, BCs_frt, BCs_Subdom,
    Glob_Mat, GAUSSQUAD, Eqn_Subdom, EqnBC_frt, SDE_sigma, device, noise, signal
):
    """
    Assemble and extract the global solution for a pure diffusion test case with fractures using PyTorch.

    Parameters:
        ny (int): Number of y-direction nodes or similar parameter.
        node_frt (int): Number of fracture nodes.
        TriMesh (list of objects): List of two TriMesh objects, each with required attributes.
        nt (int): Number of time steps.
        dt (float): Time step size.
        poros_subdom (float): Porosity for subdomains.
        poros_frt (float): Porosity for fractures.
        Glob_Mat (list of PyTorch sparse tensors): List of two global matrices, one per subdomain.
        GAUSSQUAD (dict): Dictionary containing quadrature points and weights.
        EqnBC_fracture (list of objects): List of two boundary condition objects for fractures.

    Returns:
        Glob_Sln (torch.Tensor): Assembled global solution matrix.
    """
    # Initialize lists for boundary edges and equation boundary conditions
    FractureEgs = [None] * 2
    DirichletEgs = [None] * 2
    NeumannEgs = [None] * 2
    EqnBC = [None] * 2
    
    for k in range(2):
        FractureEgs[k] = TriMesh[k].FractureEgs
        DirichletEgs[k] = TriMesh[k].DirichletEgs
        NeumannEgs[k] = TriMesh[k].NeumannEgs
        EqnBC[k]  = Eqn_Subdom[k]
        
    node = [None] * 2
    elem = [None] * 2
    edge = [None] * 2
    DOFs_Local = [None] * 2

    for k in range(2):
        node[k] = TriMesh[k].node
        elem[k] = TriMesh[k].elem
        edge[k] = TriMesh[k].edge
        DOFs_Local[k] = TriMesh[k].DOFs
    
#     print(DirichletEgs[0].shape)
    presBC_subdom = [torch.zeros(DirichletEgs[k].shape[1], 1, dtype=torch.float64, device=device) for k in range(2)]
    
    # Initial conditions
    m, ensemble_size = X0.shape
    X = torch.zeros_like(X0)
#     noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(X0)
    c0 = [None, None]
    
    c0[0] = X0[TriMesh[0].NumEgs + torch.arange(0, TriMesh[0].NumEms), :]
    c0[1] = X0[DOFs_Local[0] + TriMesh[1].NumEgs + torch.arange(0, TriMesh[1].NumEms), :]
    c0_frt = X0[DOFs_Local[0] + DOFs_Local[1] + node_frt + torch.arange(0, ny), :]
    
    C = torch.cat((c0[0], c0[1], c0_frt), dim=0)
    if signal == 0:
        noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(C)
          
    pres_top_frt = BCs_frt
    presBC_subdom[0][:, 0] = 0
    
#     for n in range(nt):  
    pres_bot_frt = 0
    presBC_subdom[1][:, 0] += BCs_Subdom

    # Solving
    GlbRHS, flag, sln = Assemble_RHS_PureDiff_DA_General_Aniso(
        c0, c0_frt, ensemble_size, DOFs_Local, ny, DirichletEgs, NeumannEgs, FractureEgs,
        EqnBC, EqnBC_frt, TriMesh, GAUSSQUAD, Glob_Mat, dt, pres_bot_frt, pres_top_frt, 
        presBC_subdom, poros_frt, poros_subdom, noise, device
    )
    flag_flat = flag.view(-1).bool()
    EmFreeEg = torch.where(~flag_flat)[0]

    if Glob_Mat.device.type == 'cuda':
        Glob_Mat_cpu = (Glob_Mat.to_dense()).cpu()
    else:
        Glob_Mat_cpu = Glob_Mat.to_dense()

    Glob_Mat_scipy = scipy.sparse.csr_matrix(Glob_Mat_cpu.numpy())  # Or use sparse conversion

    if EmFreeEg.device.type == 'cuda':
        EmFreeEg_np = np.array(EmFreeEg.cpu().numpy())
    else:
        EmFreeEg_np = np.array(EmFreeEg.numpy())

    if GlbRHS.device.type =='cuda':
        GlbRHS_cpu = GlbRHS.cpu()
    else: 
        GlbRHS_cpu = GlbRHS

    # Solve using SciPy's sparse solver
    result = scipy.sparse.linalg.spsolve(Glob_Mat_scipy[np.ix_(EmFreeEg_np, EmFreeEg_np)], GlbRHS_cpu[EmFreeEg_np, :].numpy())

    # Convert result back to PyTorch tensor
    result = torch.from_numpy(result)
#         result = torch.linalg.solve(Glob_Mat[EmFreeEg][:, EmFreeEg], GlbRHS[EmFreeEg])
#         sln[EmFreeEg, :] = result.view(-1, 1)
#         sln[EmFreeEg, :] = result
    result = result.to(device)
    if result.dim() == 1:  # Check if result is 1D
        sln[EmFreeEg, :] += result.unsqueeze(1)  # Reshape to [4009, 1]
    else:  # result is already 2D
        sln[EmFreeEg, :] += result  # Assign directly
#         X += sln
    return sln
    
def ComputSol_DFFracture_Afrt_Ksubdom_DF_General(
    ny, node_frt, TriMesh, n_dim, nt, dt, poros_subdom, poros_frt, BCs_frt, BCs_Subdom,
    Glob_Mat, GAUSSQUAD, Eqn_Subdom, EqnBC_frt, SDE_sigma, device, noise, signal
):
    """
    Assemble and extract the global solution for a pure diffusion test case with fractures using PyTorch.

    Parameters:
        ny (int): Number of y-direction nodes or similar parameter.
        node_frt (int): Number of fracture nodes.
        TriMesh (list of objects): List of two TriMesh objects, each with required attributes.
        nt (int): Number of time steps.
        dt (float): Time step size.
        poros_subdom (float): Porosity for subdomains.
        poros_frt (float): Porosity for fractures.
        Glob_Mat (list of PyTorch sparse tensors): List of two global matrices, one per subdomain.
        GAUSSQUAD (dict): Dictionary containing quadrature points and weights.
        EqnBC_fracture (list of objects): List of two boundary condition objects for fractures.

    Returns:
        Glob_Sln (torch.Tensor): Assembled global solution matrix.
    """
    # Initialize lists for boundary edges and equation boundary conditions
    FractureEgs = [None] * 2
    DirichletEgs = [None] * 2
    NeumannEgs = [None] * 2
    EqnBC = [None] * 2
    
    for k in range(2):
        FractureEgs[k] = TriMesh[k].FractureEgs
        DirichletEgs[k] = TriMesh[k].DirichletEgs
        NeumannEgs[k] = TriMesh[k].NeumannEgs
        EqnBC[k]  = Eqn_Subdom[k]
        
    node = [None] * 2
    elem = [None] * 2
    edge = [None] * 2
    DOFs_Local = [None] * 2

    for k in range(2):
        node[k] = TriMesh[k].node
        elem[k] = TriMesh[k].elem
        edge[k] = TriMesh[k].edge
        DOFs_Local[k] = TriMesh[k].DOFs
    
#     print(DirichletEgs[0].shape)
    presBC_subdom = [torch.zeros(DirichletEgs[k].shape[1], 1, dtype=torch.float64, device=device) for k in range(2)]
    
    # Initial conditions
    # m, ensemble_size = X0.shape
    # X = torch.zeros_like(X0)
#     noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(X0)
    X0 = torch.zeros(n_dim, 1, dtype=torch.float64)
    m, ensemble_size = X0.shape
    c0 = [None, None]
    
    c0[0] = X0[TriMesh[0].NumEgs + torch.arange(0, TriMesh[0].NumEms), :]
    c0[1] = X0[DOFs_Local[0] + TriMesh[1].NumEgs + torch.arange(0, TriMesh[1].NumEms), :]
    c0_frt = X0[DOFs_Local[0] + DOFs_Local[1] + node_frt + torch.arange(0, ny), :]
    
    C = torch.cat((c0[0], c0[1], c0_frt), dim=0)
    if signal == 0:
        noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(C)
          
    pres_top_frt = 0
    presBC_subdom[0][:, 0] = 0
    
#     for n in range(nt):  
    pres_bot_frt = BCs_frt
    presBC_subdom[1][:, 0] += BCs_Subdom

    DFSln = torch.zeros(n_dim, nt+1, dtype=torch.float64)
    # Solving
    for ll in range(0, nt):
        GlbRHS, flag, sln = Assemble_RHS_PureDiff_DA_General(
            c0, c0_frt, ensemble_size, DOFs_Local, ny, DirichletEgs, NeumannEgs, FractureEgs,
            EqnBC, EqnBC_frt, TriMesh, GAUSSQUAD, Glob_Mat, dt, pres_bot_frt, pres_top_frt, 
            presBC_subdom, poros_frt, poros_subdom, noise, device
        )
        flag_flat = flag.view(-1).bool()
        EmFreeEg = torch.where(~flag_flat)[0]
    
        if Glob_Mat.device.type == 'cuda':
            Glob_Mat_cpu = (Glob_Mat.to_dense()).cpu()
        else:
            Glob_Mat_cpu = Glob_Mat.to_dense()
    
        Glob_Mat_scipy = scipy.sparse.csr_matrix(Glob_Mat_cpu.numpy())  # Or use sparse conversion
    
        if EmFreeEg.device.type == 'cuda':
            EmFreeEg_np = np.array(EmFreeEg.cpu().numpy())
        else:
            EmFreeEg_np = np.array(EmFreeEg.numpy())
    
        if GlbRHS.device.type =='cuda':
            GlbRHS_cpu = GlbRHS.cpu()
        else: 
            GlbRHS_cpu = GlbRHS
    
        # Solve using SciPy's sparse solver
        result = scipy.sparse.linalg.spsolve(Glob_Mat_scipy[np.ix_(EmFreeEg_np, EmFreeEg_np)], GlbRHS_cpu[EmFreeEg_np, :].numpy())
    
        # Convert result back to PyTorch tensor
        result = torch.from_numpy(result)
        result = result.to(device)
        if result.dim() == 1:  # Check if result is 1D
            sln[EmFreeEg, :] += result.unsqueeze(1)  # Reshape to [4009, 1]
        else:  # result is already 2D
            sln[EmFreeEg, :] += result  # Assign directly
            
        DFSln[:, ll+1] += sln[:, :].squeeze(-1)
        c0[0] = sln[TriMesh[0].NumEgs + torch.arange(0, TriMesh[0].NumEms), :]
        c0[1] = sln[DOFs_Local[0] + TriMesh[1].NumEgs + torch.arange(0, TriMesh[1].NumEms), :]
        c0_frt = sln[DOFs_Local[0] + DOFs_Local[1] + node_frt + torch.arange(0, ny), :]
        
    return DFSln

def ComputSol_DFFracture_Afrt_Ksubdom_DA_Rearrange(
    X0, ny, node_frt, TriMesh, nt, dt, poros_subdom, poros_frt, BCs_frt, BCs_Subdom,
    Glob_Mat, GAUSSQUAD, Eqn_Subdom, EqnBC_frt, SDE_sigma, device, noise, signal
):
    """
    Assemble and extract the global solution for a pure diffusion test case with fractures using PyTorch.

    Parameters:
        ny (int): Number of y-direction nodes or similar parameter.
        node_frt (int): Number of fracture nodes.
        TriMesh (list of objects): List of two TriMesh objects, each with required attributes.
        nt (int): Number of time steps.
        dt (float): Time step size.
        poros_subdom (float): Porosity for subdomains.
        poros_frt (float): Porosity for fractures.
        Glob_Mat (list of PyTorch sparse tensors): List of two global matrices, one per subdomain.
        GAUSSQUAD (dict): Dictionary containing quadrature points and weights.
        EqnBC_fracture (list of objects): List of two boundary condition objects for fractures.

    Returns:
        Glob_Sln (torch.Tensor): Assembled global solution matrix.
    """
    # Initialize lists for boundary edges and equation boundary conditions
    FractureEgs = [None] * 2
    DirichletEgs = [None] * 2
    NeumannEgs = [None] * 2
    EqnBC = [None] * 2
    
    for k in range(2):
        FractureEgs[k] = TriMesh[k].FractureEgs
        DirichletEgs[k] = TriMesh[k].DirichletEgs
        NeumannEgs[k] = TriMesh[k].NeumannEgs
        EqnBC[k]  = Eqn_Subdom[k]
        
    node = [None] * 2
    elem = [None] * 2
    edge = [None] * 2
    DOFs_Local = [None] * 2

    for k in range(2):
        node[k] = TriMesh[k].node
        elem[k] = TriMesh[k].elem
        edge[k] = TriMesh[k].edge
        DOFs_Local[k] = TriMesh[k].DOFs
    
#     print(DirichletEgs[0].shape)
    presBC_subdom = [torch.zeros(DirichletEgs[k].shape[1], 1, dtype=torch.float64, device=device) for k in range(2)]
    
    # Initial conditions
    m, ensemble_size = X0.shape
    X = torch.zeros_like(X0)
#     noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(X0)
    c0 = [None, None]
    
    c0[0] = X0[TriMesh[0].NumEgs + torch.arange(0, TriMesh[0].NumEms), :]
    c0[1] = X0[DOFs_Local[0] + TriMesh[1].NumEgs + torch.arange(0, TriMesh[1].NumEms), :]
    c0_frt = X0[DOFs_Local[0] + DOFs_Local[1] + node_frt + torch.arange(0, ny), :]
    
    C = torch.cat((c0[0], c0[1], c0_frt), dim=0)
    if signal == 0:
        noise = np.sqrt(dt)*SDE_sigma*torch.zeros_like(C)
          
    pres_top_frt = 0
    presBC_subdom[0][:, 0] = 0
    
#     for n in range(nt):  
    pres_bot_frt = BCs_frt
    presBC_subdom[1][:, 0] += BCs_Subdom

    # Solving
    GlbRHS, flag, sln = Assemble_RHS_PureDiff_DA(
        c0, c0_frt, ensemble_size, DOFs_Local, ny, DirichletEgs, NeumannEgs, FractureEgs,
        EqnBC, EqnBC_frt, TriMesh, GAUSSQUAD, Glob_Mat, dt, pres_bot_frt, pres_top_frt, 
        presBC_subdom, poros_frt, poros_subdom, noise, device
    )
    flag_flat = flag.view(-1).bool()
    EmFreeEg = torch.where(~flag_flat)[0]

    if Glob_Mat.device.type == 'cuda':
        Glob_Mat_cpu = (Glob_Mat.to_dense()).cpu()
    else:
        Glob_Mat_cpu = Glob_Mat.to_dense()

    Glob_Mat_scipy = scipy.sparse.csr_matrix(Glob_Mat_cpu.numpy())  # Or use sparse conversion

    if EmFreeEg.device.type == 'cuda':
        EmFreeEg_np = np.array(EmFreeEg.cpu().numpy())
    else:
        EmFreeEg_np = np.array(EmFreeEg.numpy())

    if GlbRHS.device.type =='cuda':
        GlbRHS_cpu = GlbRHS.cpu()
    else: 
        GlbRHS_cpu = GlbRHS
#         print(EmFreeEg.numpy().shape)
#         print(Glob_Mat_scipy[np.ix_(EmFreeEg_np, EmFreeEg_np)].shape)
    # Solve using SciPy's sparse solver
    result = scipy.sparse.linalg.spsolve(Glob_Mat_scipy[np.ix_(EmFreeEg_np, EmFreeEg_np)], GlbRHS_cpu[EmFreeEg_np, :].numpy())

    # Convert result back to PyTorch tensor
    result = torch.from_numpy(result)
#         result = torch.linalg.solve(Glob_Mat[EmFreeEg][:, EmFreeEg], GlbRHS[EmFreeEg])
#         sln[EmFreeEg, :] = result.view(-1, 1)
#         sln[EmFreeEg, :] = result
    result = result.to(device)
    if result.dim() == 1:  # Check if result is 1D
        sln[EmFreeEg, :] += result.unsqueeze(1)  # Reshape to [4009, 1]
    else:  # result is already 2D
        sln[EmFreeEg, :] += result  # Assign directly
#         X += sln
    sln = RearrangeSol(sln, TriMesh)
    return sln