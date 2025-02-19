function TriMesh = RectDom_TriMesh_GenUnfm(xa,xb,nx,yc,yd,ny,status)
%% Generating a uniform triangular mesh on a given rectangular domain 
% Input:  
%   xa: left  xb: right  nx: number of partitions in the x-direction 
%   yc: bottom  yd: top  ny: number of partitions in the y-direction 
% Output: 
%   TriMesh: a structure of arrays for primary mesh info 
% JL20160106: This function is mainly for simplicity not efficiency 
% Originally authored by Rachel Cali, ColoState, Spring 2007 
% Revised by James Liu, ColoState; 2007/01--2016/09

%% For status=1
TriMesh.NumNds = (nx+1)*(ny+1);
TriMesh.NumEms = 2*nx*ny;

% Generating nodes 
x = linspace(xa,xb,nx+1);
y = linspace(yc,yd,ny+1);
[X,Y] = meshgrid(x,y);
TriMesh.node = [X(:),Y(:)];  % JL20160106: The lexicongraphical order!!

% Generating elements 
TriMesh.elem = zeros(TriMesh.NumEms,3);
for i=1:nx
  for j=1:ny
    % Left triangle, clockwise, starting at the right-angle corner 
    k = (i-1)*(2*ny)+2*j-1;
    TriMesh.elem(k,1) = (i-1)*(ny+1)+j;
    TriMesh.elem(k,2) = TriMesh.elem(k,1) + (ny+1);
    TriMesh.elem(k,3) = TriMesh.elem(k,1) + 1;
    % Right triangle, clockwise, starting at the right-angle corner 
    k = (i-1)*(2*ny)+2*j;
    TriMesh.elem(k,1) = i*(ny+1)+j+1;
    TriMesh.elem(k,2) = TriMesh.elem(k,1) - (ny+1); 
    TriMesh.elem(k,3) = TriMesh.elem(k,1) - 1;
  end
end

if (status==1)
  TriMesh.flag = 1;
  return;
end

%% For status=2 (Secondary mesh info)
TriMesh.NumEgsDiag = nx*ny;
TriMesh.NumEgsHori = nx*(ny+1);
TriMesh.NumEgsVert = (nx+1)*ny;
TriMesh.NumEgs =  TriMesh.NumEgsVert + TriMesh.NumEgsHori + TriMesh.NumEgsDiag;

%% Setting up edges 
TriMesh.edge = zeros(TriMesh.NumEgs,2);
% Vertical edges 
for i=0:nx
  for j=1:ny
    k = i*ny+j;
    TriMesh.edge(k,1:2) = [i*(ny+1)+j, i*(ny+1)+j+1];
  end
end
% Horizontal edges 
for i=1:nx
  for j=0:ny
    k = TriMesh.NumEgsVert + j*nx+i;
    TriMesh.edge(k,1:2) = [(i-1)*(ny+1)+j+1, i*(ny+1)+j+1];
  end
end
% Diagonal edges 
for i=1:nx
  for j=1:ny
    k = TriMesh.NumEgsVert + TriMesh.NumEgsHori + (i-1)*ny+j;
    TriMesh.edge(k,1:2) = [(i-1)*(ny+1)+j+1, i*(ny+1)+j];
  end
end

% Generating secondary mesh info on element-vs-edges 
% disp('Setting up element-vs-edges...'); 
TriMesh.elem2edge = zeros(TriMesh.NumEms,3);
for i=1:nx
  for j=1:ny
    % Left triangle 
    k = (i-1)*(2*ny)+2*j-1;
    TriMesh.elem2edge(k,1) = TriMesh.NumEgsVert+TriMesh.NumEgsHori+...
        (i-1)*ny+j;  % diag.
    TriMesh.elem2edge(k,2) = (i-1)*ny+j;                            % vert.
    TriMesh.elem2edge(k,3) = TriMesh.NumEgsVert+(j-1)*nx+i;               % hori.
    % Right triangle 
    k = (i-1)*(2*ny)+2*j;
    TriMesh.elem2edge(k,1) = TriMesh.NumEgsVert + TriMesh.NumEgsHori + (i-1)*ny+j;  % diag.
    TriMesh.elem2edge(k,2) = i*ny+j;                                % vert.
    TriMesh.elem2edge(k,3) = TriMesh.NumEgsVert + j*nx+i;                   % hori.
  end
end

% Generating secondary mesh info on edge-vs-elements based on elem2edge 
TriMesh.edge2elem = zeros(TriMesh.NumEgs,2);
CntEmsEg = zeros(TriMesh.NumEgs,1);
for ie=1:TriMesh.NumEms
    LblEg = TriMesh.elem2edge(ie,1:3);
    CntEmsEg(LblEg) = CntEmsEg(LblEg) + 1;
    for k=1:3
        TriMesh.edge2elem(LblEg(k),CntEmsEg(LblEg(k))) = ie;
    end
end

% Adjusting 
% for ig=1:TriMesh.NumEgs
%   if TriMesh.edge2elem(ig,1)>TriMesh.edge2elem(ig,2)
%     tmp = TriMesh.edge2elem(ig,1);
%     TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
%     TriMesh.edge2elem(ig,2) = tmp;
%   end
% end
ig = find(TriMesh.edge2elem(:,1)>TriMesh.edge2elem(:,2));
tmp = TriMesh.edge2elem(ig,1);
TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
TriMesh.edge2elem(ig,2) = tmp;
% for ig=1:TriMesh.NumEgs
%   if TriMesh.edge2elem(ig,1)==0
%     TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
%     TriMesh.edge2elem(ig,2) = 0;
%   end
% end
ig = find(TriMesh.edge2elem(:,1)==0);
TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
TriMesh.edge2elem(ig,2) = 0;

%% Generating secondary mesh info on element areas and edge lengths 
% areas for all elements 
k1 = TriMesh.elem(:,1);  k2 = TriMesh.elem(:,2);  k3 = TriMesh.elem(:,3);
x1 = TriMesh.node(k1,1);  y1 = TriMesh.node(k1,2); 
x2 = TriMesh.node(k2,1);  y2 = TriMesh.node(k2,2);
x3 = TriMesh.node(k3,1);  y3 = TriMesh.node(k3,2);
TriMesh.area = 0.5*((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1));
% length for all edges 
k1 = TriMesh.edge(:,1);  k2 = TriMesh.edge(:,2);
x1 = TriMesh.node(k1,1);  y1 = TriMesh.node(k1,2);  
x2 = TriMesh.node(k2,1);  y2 = TriMesh.node(k2,2);  
TriMesh.LenEg = sqrt((x2-x1).^2+(y2-y1).^2);

%% Finishing secondary mesh info
TriMesh.flag = 2;

%% Afternotes 
% Diagonal edges: "backslashing" 
% Formula for node label 
%   pos=(i,j) i[0,nx],j[0,ny], label=i*(ny+1)+j+1;
% Formula for element label 
%     pos=(i,j): i[1,nx],j[1,ny], 
%    lower/left: label=(i-1)*(2*ny)+2*j-1;
%   upper/right: label=(i-1)*(2*ny)+2*j;
% Formulas for edge label within each group 
%   Vert:  pos=(i,j) i[0,nx],j[1,ny], label=i*ny+j;
%   Hori:  pos=(i,j) i[1,nx],j[0,ny], label=j*nx+i;
%   Diag:  pos=(i,j) i[1,nx],j[1,ny], label=(i-1)*ny+j;

return;