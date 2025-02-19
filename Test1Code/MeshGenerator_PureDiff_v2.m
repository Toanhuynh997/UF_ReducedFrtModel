function TriMesh = MeshGenerator_PureDiff_v2(xa, xb, yc, yd, nx, ny)

TriMesh       = cell(2, 1);
BndryDescMat  = cell(2, 1);
BndryCondType = cell(2, 1);

id_boundary   = cell(2, 1);
id_dirichlet  = cell(2, 1);

FractureEgs   = cell(2, 1);
DirichletEgs = cell(2, 1);
NeumannEgs = cell(2, 1);

DOFs      = cell(2, 1);

x_start  = cell(2,1);
x_end    = cell(2,1);
y_start  = cell(2,1);
y_end    = cell(2,1);

delta_x  = (xb-xa)/nx;

x_start{1} = xa;
x_start{2} = (nx/2)*delta_x;

x_end{1}   = x_start{2};
x_end{2}   = xb;

y_start{1} = yc;
y_start{2} = yc;

y_end{1}   = yd;
y_end{2}   = yd;

for k=1:2
    TriMesh{k} = RectDom_TriMesh_GenUnfm(x_start{k}, x_end{k}, nx/2,...
        y_start{k}, y_end{k}, ny, 2);
end

for k=1:2
    BndryDescMat{k} = [x_start{k}, y_start{k}, x_end{k}, y_start{k}, 0,-1;...
                       x_end{k}, y_start{k}, x_end{k}, y_end{k},     1, 0;... 
                       x_end{k}, y_end{k}, x_start{k}, y_end{k},     0, 1;...
                       x_start{k}, y_end{k}, x_start{k}, y_start{k},-1, 0];
end

BndryCondType{1} = [0; 2; 1; 2; 1];
BndryCondType{2} = [0; 2; 1; 2; 1];


for k=1:2
    DOFs{k} = TriMesh{k}.NumEms + TriMesh{k}.NumEgs;
end

for k=1:2
    TriMesh{k} = TriMesh_Enrich2(TriMesh{k}, BndryDescMat{k});
    TriMesh{k} = TriMesh_Enrich3(TriMesh{k}, BndryDescMat{k});
end 

EqnBC    = cell(2, 1);
EqnBC{1} = EqnBC_sub_1;
EqnBC{2} = EqnBC_sub_2;

% diffusion{1}(:) = 1;
% diffusion{2}(:) = 1;

GAUSSQUAD = SetGaussQuad(2,4,3);

PermK = cell(2, 1);
for k=1:2
    PermK{k} = Darcy_SmplnPerm_TriMesh(EqnBC{k}.fxnK, TriMesh{k}, GAUSSQUAD);
end

for k = 1:2
    TriMesh{k}.BndryDescMat =BndryDescMat{k};
    TriMesh{k}.EqnBC = EqnBC{k};
    TriMesh{k}.DOFs = DOFs{k};
    % TriMesh{k}.diffusion = diffusion{k};
    TriMesh{k}.PermK = PermK{k};
    TriMesh{k}.BndryCondType = BndryCondType{k};
end

id_boundary{1} = 2; % fracture part 
id_boundary{2} = 4; % fracture part
id_dirichlet{1} = 4; % 
id_dirichlet{2} = 2; % 

for k=1:2
     FractureEgs{k} = find(TriMesh{k}.BndryEdge==id_boundary{k});
end

for k =1:2
    TriMesh{k}.FractureEgs = FractureEgs{k};
end


edge = cell(2, 1);
node = cell(2, 1);
elem = cell(2, 1);
for k=1:2
    edge{k} = TriMesh{k}.edge;
    node{k} = TriMesh{k}.node;
    elem{k} = TriMesh{k}.elem;
end

for k = 1:2
    DirichletEgs{k} = find(TriMesh{k}.BndryEdge==id_dirichlet{k} &...
        node{k}(edge{k}(:, 2), 2) <= 0.2);
end


for k=1:2
    NeumannEgs{k} = find(TriMesh{k}.BndryEdge == 3 |...
        TriMesh{k}.BndryEdge==1 |...
        (TriMesh{k}.BndryEdge==id_dirichlet{k} &...
        node{k}(edge{k}(:, 2), 2) > 0.2));
end


for k=1:2
    TriMesh{k}.DirichletEgs = DirichletEgs{k};
    TriMesh{k}.NeumannEgs = NeumannEgs{k};
end
