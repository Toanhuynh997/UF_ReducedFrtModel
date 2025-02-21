% Object-oriented programming: define new type of object
classdef subdom
    properties
        RectMesh
        Precond
        nt
        dt
        xa
        xb
        yc
        yd
        elem_loc2glob
        elem_glob2loc
        edge_loc2glob
        edge_glob2loc
        boundary_local_id % boundary edges with local edge index
        boundary_global_id % boundary edges with glocal edge index
        bd_local % boundary edges with local boundary index [left right bottom top]
        bd_global % boundary edges with global boundary index
        neumann_bd
        if_local_id % interface edges with local edge index
        if_local_bd % interface edges with local boundary index
        if_global_id
        all_bc_Darcy
        all_bc
        p0
        p_prev
        Wloc
        Wrhs
        Wlambda
        Wtotal
        proj_loc2if
        proj_if2loc
        invKele
        invDele
        velo
        poros
        diff
        hydraulic
        ventcell_bd
        n_ori_out
    end
    
end