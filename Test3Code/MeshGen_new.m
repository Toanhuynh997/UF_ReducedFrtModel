function RectMesh = MeshGen_new(xa,xb,nx,yc,yd,ny)
%% Generating a uniform triangular mesh on a given rectangular domain 
% Input:  
%   xa: left  xb: right  nx: number of partitions in the x-direction 
%   yc: bottom  yd: top  ny: number of partitions in the y-direction 
% Output: 
%   RectMesh: a structure of arrays for primary mesh info 


%% For status=1
NumEgsHor=nx*(ny+1);
NumEgsVer=(nx+1)*ny;
RectMesh.NumEgs = NumEgsHor+NumEgsVer;
RectMesh.NumEms = nx*ny;
RectMesh.NumDofus=4*RectMesh.NumEms;

% Generating nodes 
xgrid=linspace(xa,xb,nx+1);
RectMesh.xgrid = xgrid; 
ygrid = linspace(yc,yd,ny+1);
RectMesh.ygrid=ygrid;
%[X,Y] = meshgrid(x,y);

%
%% Generating secondary mesh info on edge-vs-elements

RectMesh.edge2elem=zeros(RectMesh.NumEgs,2); % 1st and 2nd col: 2 elements sharing the edge, if on the boundary, 2nd col=-1; 
RectMesh.edgeDir=zeros(RectMesh.NumEgs,1); % direction, 2: vertical, -2: horizontal

RectMesh.NumBdEgs=2*(nx+ny);
NumBdEgs=2*(nx+ny);

%******** boundary <--> edge

%RectMesh.boundaryType=zeros(NumBdEgs,1);  % 1: Dirichlet, 2: Neumann; 3: Robin
RectMesh.boundaryPos=zeros(NumBdEgs,1); % 1: left, 2: right, 3: bottom, 4: top
RectMesh.bdLeft=zeros(ny,3); % 1st col: boundary index; 2nd & 3rd col: coordinates
RectMesh.bdRight=zeros(ny,3);
RectMesh.bdBot=zeros(nx,3);
RectMesh.bdTop=zeros(nx,3);


% boundary indices: left, right (upward), bottom, top (left to right)

% 
RectMesh.bdLeft(:,1)= 1:ny;    
RectMesh.bdRight(:,1)=ny+1:2*ny; 
RectMesh.bdBot(:,1)=2*ny+1:2*ny+nx; 
RectMesh.bdTop(:,1)=2*ny+nx+1:2*ny+2*nx;

RectMesh.bdLeft(:,2)=ygrid(1:ny);
RectMesh.bdLeft(:,3)=ygrid(2:ny+1);
RectMesh.bdRight(:,2)=ygrid(1:ny);
RectMesh.bdRight(:,3)=ygrid(2:ny+1);

RectMesh.bdBot(:,2)=xgrid(1:nx);
RectMesh.bdBot(:,3)=xgrid(2:nx+1);
RectMesh.bdTop(:,2)=xgrid(1:nx);
RectMesh.bdTop(:,3)=xgrid(2:nx+1);


RectMesh.bdLbc=zeros(ny,2); % 1st col: BCs type (1: Dirichlet, 2: Neumann; 3: Robin); 2nd: BCs data
RectMesh.bdRbc=zeros(ny,2);
RectMesh.bdBbc=zeros(nx,2);
RectMesh.bdTbc=zeros(nx,2);


vert_edge=(reshape(1:(nx+1)*ny,nx+1, ny))';       % ny x Nx matrix
horz_edge=NumEgsVer+(reshape(1:nx*(ny+1),nx, ny+1))';  % Ny x nx matrix
RectMesh.boundary2edge=zeros(NumBdEgs,1); % edge index
RectMesh.boundary2edge=[vert_edge(:,1); vert_edge(:,end);...
    (horz_edge(1,:))'; (horz_edge(end,:))'];

RectMesh.edge2boundary=zeros(RectMesh.NumEgs,1); % if internal: 0, otherwise: boundary index

for l=1:NumBdEgs
    RectMesh.edge2boundary(RectMesh.boundary2edge(l))=l;
end

% bc_count=0;
% left_ct=0;
% right_ct=0;
% bot_ct=0;
% top_ct=0;

%******** edge <--> elements

% Vertical edges
for j=1:ny
  for i=1:(nx+1)
    k  = (j-1)*(nx+1)+i;
    ie = i+(j-1)*nx;
    RectMesh.edgeDir(k)=2;
    if i==1
        RectMesh.edge2elem(k,1:2) = [ie, -1]; % -1 left boundary
%         bc_count=bc_count+1;
%         RectMesh.boundaryPos(bc_count)=1;
%         left_ct=left_ct+1;
%         RectMesh.bdLeft(left_ct,1)=bc_count;
%         RectMesh.bdLeft(left_ct,2)=ygrid(j);
%         RectMesh.bdLeft(left_ct,3)=ygrid(j+1);
%         
%         RectMesh.boundary2edge(bc_count)=k;
%         RectMesh.edge2boundary(k)=bc_count;
            
    elseif i==nx+1
        RectMesh.edge2elem(k,1:2) = [ie-1, -1]; % -1 right boundary
%         bc_count=bc_count+1;
%         RectMesh.boundaryPos(bc_count)=2;
%         right_ct=right_ct+1;
%         RectMesh.bdRight(right_ct,1)=bc_count;
%         RectMesh.bdRight(right_ct,2)=ygrid(j);
%         RectMesh.bdRight(right_ct,3)=ygrid(j+1);
%         
%         RectMesh.boundary2edge(bc_count)=k;
%         RectMesh.edge2boundary(k)=bc_count;
    else
        RectMesh.edge2elem(k,1:2) = [ie-1,ie]; % 1: left, 2: right
    end
  end
end

% Horizontal edges
for j=1:(ny+1)
   for i=1:nx
    %k=NumEgsVer+j+(i-1)*(ny+1);
    k  = NumEgsVer+(j-1)*(nx)+i;
    RectMesh.edgeDir(k)=-2;
    if j==1
        ie=i+(j-1)*nx;
        RectMesh.edge2elem(k,1:2) = [ie, -1]; % -1 bottom boundary
%         bc_count=bc_count+1;
%         RectMesh.boundaryPos(bc_count)=3;
%         bot_ct=bot_ct+1;
%         RectMesh.bdBot(bot_ct,1)=bc_count;
%         RectMesh.bdBot(bot_ct,2)=xgrid(i);
%         RectMesh.bdBot(bot_ct,3)=xgrid(i+1);
%         
%         RectMesh.boundary2edge(bc_count)=k;
%         RectMesh.edge2boundary(k)=bc_count;
    elseif j==ny+1
        ie=i+(ny-1)*nx;
        RectMesh.edge2elem(k,1:2) = [ie, -1]; % -1 top boundary
%         bc_count=bc_count+1;
%         RectMesh.boundaryPos(bc_count)=4;
%         top_ct=top_ct+1;
%         RectMesh.bdTop(top_ct,1)=bc_count;
%         RectMesh.bdTop(top_ct,2)=xgrid(i);
%         RectMesh.bdTop(top_ct,3)=xgrid(i+1);
%         
%         RectMesh.boundary2edge(bc_count)=k;
%         RectMesh.edge2boundary(k)=bc_count;
    else
        ie=i+(j-1)*nx;
        ie_nei=i+(j-2)*nx;
        RectMesh.edge2elem(k,1:2) = [ie_nei,ie]; % 1: bottom, 2: top
    end
  end
end
%
%
%% Generating secondary mesh info on element-vs-edges 
%  |-------4---------|
%  |1                |2
%  |                 |
%  |-------3---------|                
%
RectMesh.elem_xcoor=zeros(RectMesh.NumEms,2); % x_a, x_b
RectMesh.elem_ycoor=zeros(RectMesh.NumEms,2); % y_c, y_d
RectMesh.elem_hx=zeros(RectMesh.NumEms,1); % hx
RectMesh.elem_hy=zeros(RectMesh.NumEms,1); % hy
RectMesh.elem_center=zeros(RectMesh.NumEms,2); % xmid, ymid

RectMesh.elem2edge=zeros(RectMesh.NumEms,4);
RectMesh.elem2dofu=zeros(RectMesh.NumEms,4);
RectMesh.dofu2elem=zeros(RectMesh.NumDofus,1); % 1st: elem index; 2nd: 1(left), 2(right), 3(bottom), or 4(top)
RectMesh.dofu2edge=zeros(RectMesh.NumDofus,1);
RectMesh.dofu2bd=zeros(RectMesh.NumDofus,1); % bd index: boundary edge,  0: internal

RectMesh.bd2dofu=zeros(NumBdEgs,1);

for j=1:ny
    for i=1:nx
        k=i+(j-1)*nx;
        
        % Element k: [xa, xb] x [yc, yd]
        RectMesh.elem_xcoor(k,1)=xgrid(i);   % xa
        RectMesh.elem_xcoor(k,2)=xgrid(i+1); % xb
        RectMesh.elem_hx(k)=RectMesh.elem_xcoor(k,2)-RectMesh.elem_xcoor(k,1);
        RectMesh.elem_ycoor(k,1)=ygrid(j);   % yc
        RectMesh.elem_ycoor(k,2)=ygrid(j+1); % yd   
        RectMesh.elem_hy(k)=RectMesh.elem_ycoor(k,2)-RectMesh.elem_ycoor(k,1);
        RectMesh.elem_center(k,1)=(RectMesh.elem_xcoor(k,2)+RectMesh.elem_xcoor(k,1))/2; % xmid
        RectMesh.elem_center(k,2)=(RectMesh.elem_ycoor(k,2)+RectMesh.elem_ycoor(k,1))/2; % ymid
        
        edgeV=i+(j-1)*(nx+1); 
        %edgeH=NumEgsVer+j+(i-1)*(ny+1);
        edgeH= NumEgsVer+(j-1)*(nx)+i;
        RectMesh.elem2edge(k,1)=edgeV;
        RectMesh.elem2edge(k,2)=edgeV+1;
        RectMesh.elem2edge(k,3)=edgeH;
        RectMesh.elem2edge(k,4)=edgeH+nx;%edgeH+1;
        
        il=(k-1)*4;
        RectMesh.elem2dofu(k,1)=il+1;
        RectMesh.elem2dofu(k,2)=il+2;
        RectMesh.elem2dofu(k,3)=il+3;
        RectMesh.elem2dofu(k,4)=il+4;
        
        RectMesh.dofu2elem(il+1:il+4,1)=k; 
    %     RectMesh.dofu2elem(il+1,2)=1;
    %     RectMesh.dofu2elem(il+2,2)=2;
    %     RectMesh.dofu2elem(il+3,2)=3;
    %     RectMesh.dofu2elem(il+4,2)=4;
        RectMesh.dofu2edge(il+1)= edgeV; 
        RectMesh.dofu2edge(il+2)= edgeV+1;
        RectMesh.dofu2edge(il+3)= edgeH;
        RectMesh.dofu2edge(il+4)= edgeH+nx;%edgeH+1;
        
        if RectMesh.edge2elem(edgeV,2)==-1
               bd_id= RectMesh.edge2boundary(edgeV);
                RectMesh.dofu2bd(il+1)= bd_id;
                RectMesh.bd2dofu(bd_id)=il+1;
        end
    
        if RectMesh.edge2elem(edgeV+1,2)==-1
                bd_id= RectMesh.edge2boundary(edgeV+1);
                RectMesh.dofu2bd(il+2)= bd_id;
                RectMesh.bd2dofu(bd_id)=il+2;
        end
    
        if RectMesh.edge2elem(edgeH,2)==-1
                bd_id= RectMesh.edge2boundary(edgeH);
                RectMesh.dofu2bd(il+3)= bd_id;
                RectMesh.bd2dofu(bd_id)=il+3;
        end
    
        if RectMesh.edge2elem(edgeH+nx,2)==-1
                bd_id= RectMesh.edge2boundary(edgeH+nx); %edgeH+nx
                RectMesh.dofu2bd(il+4)= bd_id;
                RectMesh.bd2dofu(bd_id)=il+4;
                
        end

    end
    
    
end

RectMesh.NumIntEgs=RectMesh.NumEgs-NumBdEgs;
RectMesh.IntEdge2dofu=zeros(RectMesh.NumIntEgs,3); % dofu for left/bottom element, dofu for right/top element
RectMesh.IntEdge2Edge=zeros(RectMesh.NumEgs,1); 
RectMesh.Edge2IntEdge=zeros(RectMesh.NumEgs,1); % if internal: IntEdge, otherwise: 0
inte_count=0;

for k=1:RectMesh.NumEgs
    
    if RectMesh.edgeDir(k)==2 && RectMesh.edge2elem(k,2)~=-1 % vertical and not on boundary
        inte_count=inte_count+1;
        RectMesh.IntEdge2Edge(inte_count,1)=k;
        RectMesh.Edge2IntEdge(k)=inte_count;
        RectMesh.IntEdge2dofu(inte_count,2)=RectMesh.elem2dofu(RectMesh.edge2elem(k,1),2); % left element
        RectMesh.IntEdge2dofu(inte_count,3)=RectMesh.elem2dofu(RectMesh.edge2elem(k,2),1); % right element
    end
    
    
    if RectMesh.edgeDir(k)==-2 && RectMesh.edge2elem(k,2)~=-1 % horizontal and not on boundary
        inte_count=inte_count+1;
        RectMesh.IntEdge2Edge(inte_count,1)=k;
        RectMesh.Edge2IntEdge(k)=inte_count;
        RectMesh.IntEdge2dofu(inte_count,2)=RectMesh.elem2dofu(RectMesh.edge2elem(k,1),4); % bottom element
        RectMesh.IntEdge2dofu(inte_count,3)=RectMesh.elem2dofu(RectMesh.edge2elem(k,2),3); % top element
    end
    
end

% RectMesh.edge2dofu=zeros(RectMesh.NumEgs,2);
% 
% for k=1:RectMesh.NumEgs
%     
%     if RectMesh.edgeDir(k)==2 && RectMesh.edge2elem(k,2)~=-1 % vertical and not on boundary
%         RectMesh.edge2dofu(k,1)=RectMesh.elem2dofu(RectMesh.edge2elem(k,1),2); % left element
%         RectMesh.edge2dofu(k,2)=RectMesh.elem2dofu(RectMesh.edge2elem(k,2),1); % right element
%     end
%     
%     
%     if RectMesh.edgeDir(k)==-2 && RectMesh.edge2elem(k,2)~=-1 % horizontal and not on boundary
%         RectMesh.edge2dofu(k,1)=RectMesh.elem2dofu(RectMesh.edge2elem(k,1),4); % bottom element
%         RectMesh.edge2dofu(k,2)=RectMesh.elem2dofu(RectMesh.edge2elem(k,2),3); % top element
%     end
%     
%     if  RectMesh.edge2elem(k,2)==-1 && RectMesh.boundaryType(RectMesh.edge2boundary(k))~=2 % on boundary and not Neumann type (i.e. Dirichlet or Robin)
%         RectMesh.edge2dofu(k,1)=RectMesh.bd2dofu(RectMesh.edge2boundary(k));
%         RectMesh.edge2dofu(k,2)=-1; % no neighbor
%     end
%     
% end

%RectMesh.node = [X(:),Y(:)];  % JL20160106: The lexicongraphical order!!


% 
% Generating elements 
% TriMesh.elem = zeros(TriMesh.NumEms,3);
% for i=1:nx
%   for j=1:ny
%     Left triangle, counterclockwise, starting at the right-angle corner 
%     k = (i-1)*(2*ny)+2*j-1;
%     TriMesh.elem(k,1) = (i-1)*(ny+1)+j;
%     TriMesh.elem(k,2) = TriMesh.elem(k,1) + (ny+1);
%     TriMesh.elem(k,3) = TriMesh.elem(k,1) + 1;
%     Right triangle, counterclockwise, starting at the right-angle corner 
%     k = (i-1)*(2*ny)+2*j;
%     TriMesh.elem(k,1) = i*(ny+1)+j+1;
%     TriMesh.elem(k,2) = TriMesh.elem(k,1) - (ny+1); 
%     TriMesh.elem(k,3) = TriMesh.elem(k,1) - 1;
%   end
% end
% 
% if (status==1)
%   TriMesh.flag = 1;
%   return;
% end
% 
% % For status=2 (Secondary mesh info)
% NumEgsDiag = nx*ny;
% NumEgsHori = nx*(ny+1);
% NumEgsVert = (nx+1)*ny;
% TriMesh.NumEgs =  NumEgsVert + NumEgsHori + NumEgsDiag;
% 
% % Setting up edges 
% TriMesh.edge = zeros(TriMesh.NumEgs,2);
% Vertical edges 
% for i=0:nx
%   for j=1:ny
%     k = i*ny+j;
%     TriMesh.edge(k,1:2) = [i*(ny+1)+j, i*(ny+1)+j+1];
%   end
% end
% Horizonatl edges 
% for i=1:nx
%   for j=0:ny
%     k = NumEgsVert + j*nx+i;
%     TriMesh.edge(k,1:2) = [(i-1)*(ny+1)+j+1, i*(ny+1)+j+1];
%   end
% end
% Diagonal edges 
% for i=1:nx
%   for j=1:ny
%     k = NumEgsVert + NumEgsHori + (i-1)*ny+j;
%     TriMesh.edge(k,1:2) = [(i-1)*(ny+1)+j+1, i*(ny+1)+j];
%   end
% end
% 
% Generating secondary mesh info on element-vs-edges 
% disp('Setting up element-vs-edges...'); 
% TriMesh.elem2edge = zeros(TriMesh.NumEms,3);
% for i=1:nx
%   for j=1:ny
%     Left triangle 
%     k = (i-1)*(2*ny)+2*j-1;
%     TriMesh.elem2edge(k,1) = NumEgsVert + NumEgsHori + (i-1)*ny+j;  % diag.
%     TriMesh.elem2edge(k,2) = (i-1)*ny+j;                            % vert.
%     TriMesh.elem2edge(k,3) = NumEgsVert + (j-1)*nx+i;               % hori.
%     Right triangle 
%     k = (i-1)*(2*ny)+2*j;
%     TriMesh.elem2edge(k,1) = NumEgsVert + NumEgsHori + (i-1)*ny+j;  % diag.
%     TriMesh.elem2edge(k,2) = i*ny+j;                                % vert.
%     TriMesh.elem2edge(k,3) = NumEgsVert + j*nx+i;                   % hori.
%   end
% end
% 
% Generating secondary mesh info on edge-vs-elements based on elem2edge 
% TriMesh.edge2elem = zeros(TriMesh.NumEgs,2);
% CntEmsEg = zeros(TriMesh.NumEgs,1);
% for ie=1:TriMesh.NumEms
%   LblEg = TriMesh.elem2edge(ie,1:3);
%   CntEmsEg(LblEg) = CntEmsEg(LblEg) + 1;
%   for k=1:3
%     TriMesh.edge2elem(LblEg(k),CntEmsEg(LblEg(k))) = ie;
%   end
% end
% 
% Adjusting 
% for ig=1:TriMesh.NumEgs
%   if TriMesh.edge2elem(ig,1)>TriMesh.edge2elem(ig,2)
%     tmp = TriMesh.edge2elem(ig,1);
%     TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
%     TriMesh.edge2elem(ig,2) = tmp;
%   end
% end
% ig = find(TriMesh.edge2elem(:,1)>TriMesh.edge2elem(:,2));
% tmp = TriMesh.edge2elem(ig,1);
% TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
% TriMesh.edge2elem(ig,2) = tmp;
% for ig=1:TriMesh.NumEgs
%   if TriMesh.edge2elem(ig,1)==0
%     TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
%     TriMesh.edge2elem(ig,2) = 0;
%   end
% end
% ig = find(TriMesh.edge2elem(:,1)==0);
% TriMesh.edge2elem(ig,1) = TriMesh.edge2elem(ig,2);
% TriMesh.edge2elem(ig,2) = 0;
% 
% % Generating secondary mesh info on element areas and edge lengths 
% areas for all elements 
% k1 = TriMesh.elem(:,1);  k2 = TriMesh.elem(:,2);  k3 = TriMesh.elem(:,3);
% x1 = TriMesh.node(k1,1);  y1 = TriMesh.node(k1,2); 
% x2 = TriMesh.node(k2,1);  y2 = TriMesh.node(k2,2);
% x3 = TriMesh.node(k3,1);  y3 = TriMesh.node(k3,2);
% TriMesh.area = 0.5*((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1));
% length for all edges 
% k1 = TriMesh.edge(:,1);  k2 = TriMesh.edge(:,2);
% x1 = TriMesh.node(k1,1);  y1 = TriMesh.node(k1,2);  
% x2 = TriMesh.node(k2,1);  y2 = TriMesh.node(k2,2);  
% TriMesh.LenEg = sqrt((x2-x1).^2+(y2-y1).^2);
% 
% % Finishing secondary mesh info
% TriMesh.flag = 2;

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