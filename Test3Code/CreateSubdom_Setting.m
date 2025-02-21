function subdomain = CreateSubdom_Setting(nx, ny, NPX, NPY)


global xa xb yc yd t0 T delta

xa=0; xb=2;
yc=0; yd=1;

len_edge_frac = (yd-yc)/ny;
dy  = len_edge_frac;

diff_1 = 3.15e-04;
diff_2 = 3.15e-04 ;
diff_gamma = 9.92e-03;

diff   = [diff_1; diff_2];

hydraulic_1 = 9.92e-06;
hydraulic_2 = 9.92e-06;
hydraulic_gamma = 3.15e-05;

hydraulic = [hydraulic_1; hydraulic_2];

hydraulic1d = hydraulic_gamma*delta;
nSub=NPX*NPY;

% nt_all=ones(nSub,1)*nt;

xgrid = cell(2, 1);
ygrid = cell(2, 1);

xgrid{1} = linspace(xa,(xa+xb)/2,nx+1);
xgrid{2} = linspace((xa+xb)/2, xb, nx+1); 
for k = 1:nSub
    ygrid{k} = linspace(yc,yd,ny+1);
end

xmid = cell(2, 1);
ymid = cell(2, 1);
for k =1:nSub
    xmid{k}=(xgrid{k}(1:end-1)+xgrid{k}(2:end))/2;
    ymid{k}=(ygrid{k}(1:end-1)+ygrid{k}(2:end))/2;
end

Xmid = cell(2, 1);
Ymid = cell(2, 1);
for k =1:2
    [Xmid{k},Ymid{k}]=meshgrid(xmid{k}, ymid{k});
end

gnx=nx*NPX; 
gny=ny*NPY;
% Mesh generation
xgrid_all = linspace(xa, xb,gnx+1);

ygrid_all = linspace(yc,yd, gny+1);



xmid_all = (xgrid_all(1:end-1)+xgrid_all(2:end))/2;
ymid_all = (ygrid_all(1:end-1)+ygrid_all(2:end))/2;


[Xmid_all,Ymid_all]=meshgrid(xmid_all, ymid_all);

gRectMesh = MeshGen_new(xa,xb,gnx,yc,yd,gny);

% Generate diffusion per element

invKele_m=zeros(gny, gnx);
invKele_m(:,1:gnx/2)=1/hydraulic_1;
invKele_m(:,gnx/2+(1:end))=1/hydraulic_2;
invKele=reshape(invKele_m',[],1);

invDele_m=zeros(gny, gnx);
invDele_m(:,1:gnx/2)=1/diff_1;
invDele_m(:,gnx/2+(1:end))=1/diff_2;
invDele = reshape(invDele_m',[],1);


% porosity
poros = cell(3, 1);
poros{1} = 0.05;
poros{2} = 0.05;
poros{3} = 0.1; % fracture poros

poros_frac = poros{3};

[omega, subdomain]=stripdecomposition_advdiff(nx, ny, NPX, NPY,...
    invKele, invDele, poros, diff, hydraulic);


for k=1:nSub
    % subdomain{k}.nt=nt_all(k);
    % subdomain{k}.dt=(T-t0)/subdomain{k}.nt;
    
    % local mesh
    subdomain{k}.RectMesh= MeshGen_new(subdomain{k}.xa, subdomain{k}.xb, nx, ...
        subdomain{k}.yc, subdomain{k}.yd,ny);
end

n_ori = cell(2, 1);
for k = 1:2
    n_ori{k}= subdomain{k}.RectMesh.NumDofus+...
        subdomain{k}.RectMesh.NumEms+ subdomain{k}.RectMesh.NumIntEgs;
end

n_ori_out = cell(2, 1);
for k = 1:2
    n_ori_out{k}= n_ori{k} + subdomain{k}.RectMesh.NumBdEgs- ny;
end

for k = 1:2
    subdomain{k}.n_ori_out = n_ori_out{k};
end