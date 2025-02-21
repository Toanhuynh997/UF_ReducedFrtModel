function [omega, subdomain]=stripdecomposition_advdiff(nx,ny,NPX, NPY,...
    invKele, invDele, poros, diff, hydraulic)

% nx, ny: number of grid intervals per subdomain
%
% gnx, gny: global grid intervals
global xa xb yc yd

xa=0; xb=2;
yc=0; yd=1;

gnx=nx*NPX; 
gny=ny*NPY;
gNx=gnx+1;
gNy=gny+1;

Nx=nx+1;
Ny=ny+1;

count_gele=1; % elem global index of left bottom corner of each subdomain
count_gvert=1;
count_ghorz=gNx*gny+1;

% subdomain boundary indices: left, right, bottom, top
vert_edge=(reshape(1:Nx*ny,Nx, ny))';       % ny x Nx matrix
horz_edge=Nx*ny+(reshape(1:nx*Ny,nx, Ny))';  % Ny x nx matrix
left= vert_edge(:,1);     % ny x 1
right=vert_edge(:,end); 
bottom=(horz_edge(1,:))'; % nx x 1
top=(horz_edge(end,:))';

% % global element indices
% Um=(reshape(1:nx*ny,Nx, Ny))';

% subdomains
nSub=NPX*NPY;
Xgrid=linspace(xa,xb,NPX+1);
Ygrid=linspace(yc,yd,NPY+1);

subdomain=cell(nSub,1);
omega = domain_new;

count_if=0; % interface edge numbering

if NPY==1 && NPX>1 % vertical decomposition
    omega.if_size=ny*(NPX-1);
    omega.if_local_to_global=zeros(omega.if_size,1);
    omega.if_global_to_local=sparse(gNx*gny+gNy*gnx,1);
   
    for k=1:NPX
        subdomain{k}=subdom;
        subdomain{k}.xa=Xgrid(k);
        subdomain{k}.xb=Xgrid(k+1);
        subdomain{k}.yc=Ygrid(1);
        subdomain{k}.yd=Ygrid(2);
        subdomain{k}.elem_loc2glob=zeros(nx*ny,1); % element indices
        subdomain{k}.elem_glob2loc=sparse(gnx*gny,1);
        
        subdomain{k}.edge_loc2glob=zeros(Nx*ny+nx*Ny,1); % edge indices
        subdomain{k}.edge_glob2loc=sparse(gNx*gny+gnx*gNy,1);
        
        count_lele=0;
        count_lvert=0;
        count_lhorz=Nx*ny;
       for j1=1:ny
          % elements
          istart=count_gele+(j1-1)*gnx;
          subdomain{k}.elem_loc2glob(count_lele+1:count_lele+nx)=...
              istart:istart+nx-1;
          subdomain{k}.elem_glob2loc(istart:istart+nx-1)=...
              count_lele+1:count_lele+nx;
          count_lele=count_lele+nx;
          
          % vertical edges
          istart_vert=count_gvert+(j1-1)*gNx;
          subdomain{k}.edge_loc2glob(count_lvert+1:count_lvert+Nx)=...
              istart_vert:istart_vert+Nx-1;
          subdomain{k}.edge_glob2loc(istart_vert:istart_vert+Nx-1)=...
              count_lvert+1:count_lvert+Nx;
          count_lvert=count_lvert+Nx;
              
          % horizontal edges
          istart_horz=count_ghorz+(j1-1)*gnx;
          subdomain{k}.edge_loc2glob(count_lhorz+1:count_lhorz+nx)=....
              istart_horz:istart_horz+nx-1;
          subdomain{k}.edge_glob2loc(istart_horz:istart_horz+nx-1)=....
              count_lhorz+1:count_lhorz+nx;
          count_lhorz=count_lhorz+nx;
              
       end
       
       istart_horz=count_ghorz+ny*gnx;
       subdomain{k}.edge_loc2glob(count_lhorz+1:count_lhorz+nx)=istart_horz:istart_horz+nx-1;
       subdomain{k}.edge_glob2loc(istart_horz:istart_horz+nx-1)=count_lhorz+1:count_lhorz+nx;
       
              
       count_gele=count_gele+nx;
       count_gvert=count_gvert+nx;
       count_ghorz=count_ghorz+nx;
       
       
       if k==1 % first subdomain (left)
           subdomain{k}.boundary_local_id=zeros(2*nx+ny,1);
           subdomain{k}.boundary_global_id=zeros(2*nx+ny,1);
           
           local_bd=[left; bottom; top]; % boundary with edge index
           
           subdomain{k}.boundary_local_id=local_bd;
           subdomain{k}.boundary_global_id=subdomain{k}.edge_loc2glob(local_bd);
           
           subdomain{k}.if_local_id=zeros(ny,1);
           subdomain{k}.if_global_id=zeros(ny,1);
           
           subdomain{k}.if_local_id=right;
           subdomain{k}.if_global_id=subdomain{k}.edge_loc2glob(right);
           
           omega.if_local_to_global(count_if+1:count_if+ny)=subdomain{k}.if_global_id;
           omega.if_global_to_local(subdomain{k}.if_global_id)=count_if+1:count_if+ny;
           count_if=count_if+ny;
           
           %subdomain{k}.precondtype='firstsub';
           
       elseif k==NPX % last subdomain (right)
           
           subdomain{k}.boundary_local_id=zeros(2*nx+ny,1);
           subdomain{k}.boundary_global_id=zeros(2*nx+ny,1);
           
           local_bd=[right ;bottom; top];
           
           subdomain{k}.boundary_local_id=local_bd;
           subdomain{k}.boundary_global_id=subdomain{k}.edge_loc2glob(local_bd);
           
           subdomain{k}.if_local_id=zeros(ny,1);
           subdomain{k}.if_global_id=zeros(ny,1);
           
           subdomain{k}.if_local_id=left;
           subdomain{k}.if_global_id=subdomain{k}.edge_loc2glob(left);
           
           %subdomain{k}.precondtype='lastsub';
           
       else % internal subdomains
           
           subdomain{k}.boundary_local_id=zeros(2*nx,1);
           subdomain{k}.boundary_global_id=zeros(2*nx,1);
           
           local_bd=[bottom; top];
           
           subdomain{k}.boundary_local_id=local_bd;
           subdomain{k}.boundary_global_id=subdomain{k}.edge_loc2glob(local_bd);
           
           subdomain{k}.if_local_id=zeros(2*ny,1);
           subdomain{k}.if_global_id=zeros(2*ny,1);
           
           % right interface first
           subdomain{k}.if_local_id(1:ny)=right;
           subdomain{k}.if_global_id(1:ny)=subdomain{k}.edge_loc2glob(right);
           
           omega.if_local_to_global(count_if+1:count_if+ny)=subdomain{k}.if_global_id(1:ny);
           omega.if_global_to_local(subdomain{k}.if_global_id(1:ny))=count_if+1:count_if+ny;
           count_if=count_if+ny;
           
           % then left interface (not duplicate as it's already counted for
           % the global interface)
           subdomain{k}.if_local_id(ny+1:2*ny)=left;
           subdomain{k}.if_global_id(ny+1:2*ny)=subdomain{k}.edge_loc2glob(left);
           
           %subdomain{k}.precondtype='midsub';
           
           
       end
       
       subdomain{k}.invKele=zeros(nx*ny,1);
       subdomain{k}.invKele=invKele(subdomain{k}.elem_loc2glob);
       
       subdomain{k}.invDele=zeros(nx*ny,1);
       subdomain{k}.invDele=invDele(subdomain{k}.elem_loc2glob);
  
       subdomain{k}.poros = poros{k};
       subdomain{k}.diff = diff(k);
       subdomain{k}.hydraulic = hydraulic(k);
    end
    
    
   
end


% if NPX==1 && NPY>1 % horizontal decomposition
%     omega.if_size=Nx*(NPY-1);
%     omega.if_local_to_global=zeros(omega.if_size,1);
%     omega.if_signx=sparse(omega.if_size,1);
%     omega.if_signy=sparse(omega.if_size,1);
%     omega.if_global_to_local=sparse(gnx*gny,1);
%     for k=1:NPY
%        subdomain{k}=subdom;
%        subdomain{k}.local_to_global=zeros(Nx*Ny,1);
%        subdomain{k}.global_to_local=sparse(gnx*gny,1);
%        subdomain{k}.local_to_global=count_gele:count_gele+Nx*Ny-1;
%        subdomain{k}.global_to_local(count_gele:count_gele+Nx*Ny-1)=1:Nx*Ny;
%        count_gele=count_gele+Nx*(Ny-1);
%        
%        if k==1 % first subdomain (bottom)
%            subdomain{k}.boundary_local_id=zeros(2*Ny+Nx-2,1);
%            subdomain{k}.boundary_global_id=zeros(2*Ny+Nx-2,1);
%            subdomain{k}.boundary_signx=sparse(2*Ny+Nx-2,1);
%            subdomain{k}.boundary_signy=sparse(2*Ny+Nx-2,1);
%             
%            subdomain{k}.boundary_local_id(1:Ny)=left;
%            subdomain{k}.boundary_global_id(1:Ny)=subdomain{k}.local_to_global(left);
%            subdomain{k}.boundary_signx(1:Ny)=-1;
%            
%            subdomain{k}.boundary_local_id(Ny:Nx+Ny-1)=bottom;
%            subdomain{k}.boundary_global_id(Ny:Nx+Ny-1)=subdomain{k}.local_to_global(bottom);
%            subdomain{k}.boundary_signy(Ny:Nx+Ny-1)=-1;
%            
%            subdomain{k}.boundary_local_id(Nx+Ny-1:end)=right;
%            subdomain{k}.boundary_global_id(Nx+Ny-1:end)=subdomain{k}.local_to_global(right);
%            subdomain{k}.boundary_signx(Nx+Ny-1:end)=1;
%            
%            subdomain{k}.if_local_id=zeros(Nx,1);
%            subdomain{k}.if_global_id=zeros(Nx,1);
%            subdomain{k}.if_signx=sparse(Nx,1);
%            subdomain{k}.if_signy=sparse(Nx,1);
%            
%            subdomain{k}.if_local_id(1:Nx)=top;
%            subdomain{k}.if_global_id(1:Nx)=subdomain{k}.local_to_global(top);
%            subdomain{k}.if_signy(1:Nx)=1;
%            
%            omega.if_local_to_global(count_if+1:count_if+Nx)=subdomain{k}.if_global_id(1:Nx);
%            omega.if_signy(count_if+1:count_if+Nx)=1;
%            omega.if_global_to_local(subdomain{k}.if_global_id(1:Nx))=count_if+1:count_if+Nx;
%            count_if=count_if+Nx;
%            
%            subdomain{k}.precondtype='firstsub';
%            
%        elseif k==NPY % last subdomain (top)
%            
%            subdomain{k}.boundary_local_id=zeros(2*Ny+Nx-2,1);
%            subdomain{k}.boundary_global_id=zeros(2*Ny+Nx-2,1);
%            subdomain{k}.boundary_signx=sparse(2*Ny+Nx-2,1);
%            subdomain{k}.boundary_signy=sparse(2*Ny+Nx-2,1);
%             
%            subdomain{k}.boundary_local_id(1:Ny)=right;
%            subdomain{k}.boundary_global_id(1:Ny)=subdomain{k}.local_to_global(right);
%            subdomain{k}.boundary_signx(1:Ny)=1;
%            
%            subdomain{k}.boundary_local_id(Ny:Nx+Ny-1)=top;
%            subdomain{k}.boundary_global_id(Ny:Nx+Ny-1)=subdomain{k}.local_to_global(top);
%            subdomain{k}.boundary_signy(Ny:Nx+Ny-1)=1;
%            
%            subdomain{k}.boundary_local_id(Nx+Ny-1:end)=left;
%            subdomain{k}.boundary_global_id(Nx+Ny-1:end)=subdomain{k}.local_to_global(left);
%            subdomain{k}.boundary_signx(Nx+Ny-1:end)=-1;
%            
%            subdomain{k}.if_local_id=zeros(Nx,1);
%            subdomain{k}.if_global_id=zeros(Nx,1);
%            subdomain{k}.if_signx=sparse(Nx,1);
%            subdomain{k}.if_signy=sparse(Nx,1);
%            
%            subdomain{k}.if_local_id(1:Nx)=bottom;
%            subdomain{k}.if_global_id(1:Nx)=subdomain{k}.local_to_global(bottom);
%            subdomain{k}.if_signy(1:Nx)=-1;
%            
%            subdomain{k}.precondtype='lastsub';
%            
%            
%        else % internal subdomains
%            
%            subdomain{k}.boundary_local_id=zeros(2*Ny,1);
%            subdomain{k}.boundary_global_id=zeros(2*Ny,1);
%            subdomain{k}.boundary_signx=sparse(2*Ny,1);
%            subdomain{k}.boundary_signy=sparse(2*Ny,1);
%             
%            subdomain{k}.boundary_local_id(1:Ny)=right;
%            subdomain{k}.boundary_global_id(1:Ny)=subdomain{k}.local_to_global(right);
%            subdomain{k}.boundary_signx(1:Ny)=1;
%            
%            subdomain{k}.boundary_local_id(Ny+1:end)=left;
%            subdomain{k}.boundary_global_id(Ny+1:end)=subdomain{k}.local_to_global(left);
%            subdomain{k}.boundary_signx(Ny+1:end)=-1;
%            
%            
%            subdomain{k}.if_local_id=zeros(2*Nx,1);
%            subdomain{k}.if_global_id=zeros(2*Nx,1);
%            subdomain{k}.if_signx=sparse(2*Nx,1);
%            subdomain{k}.if_signy=sparse(2*Nx,1);
%            
%            subdomain{k}.if_local_id(1:Nx)=bottom;
%            subdomain{k}.if_global_id(1:Nx)=subdomain{k}.local_to_global(bottom);
%            subdomain{k}.if_signy(1:Nx)=-1;
%            
%            subdomain{k}.if_local_id(Nx+1:end)=top;
%            subdomain{k}.if_global_id(Nx+1:end)=subdomain{k}.local_to_global(top);
%            subdomain{k}.if_signy(Nx+1:end)=1;
%            
%            omega.if_local_to_global(count_if+1:count_if+Nx)=subdomain{k}.if_global_id(Nx+1:end);
%            omega.if_signy(count_if+1:count_if+Nx)=1;
%            omega.if_global_to_local(subdomain{k}.if_global_id(Nx+1:end))=count_if+1:count_if+Nx;
%            count_if=count_if+Nx;
%            
%            subdomain{k}.precondtype='midsub';
%            
%        end
%        
%        
%     end
% end
% 
% 
% omega.if_indx=find(omega.if_signx==1);
% 
% omega.if_indy=find(omega.if_signy==1);
% 
% 
% omega.localNx=Nx;
% omega.localNy=Ny;
% 
% 











end