function PermK = Darcy_SmplnPerm_TriMesh(fxnK,TriMesh,GAUSSQUAD)
%% Darcy a triangular mesh: Sampling permeability using a Gaussian quadrature
% PermK will be an elementwise constant 2x2 SPD matrix (as averages on elements) 
% James Liu, ColoState; 2012/07--2017/02 

% Basic mesh info 
NumEms = TriMesh.NumEms;

%% 
PermK = zeros(NumEms,2,2);
NumQuadPts = size(GAUSSQUAD.TRIG,1);
for k=1:NumQuadPts
  qp = GAUSSQUAD.TRIG(k,1) * TriMesh.node(TriMesh.elem(:,1),:) ...
     + GAUSSQUAD.TRIG(k,2) * TriMesh.node(TriMesh.elem(:,2),:) ...
     + GAUSSQUAD.TRIG(k,3) * TriMesh.node(TriMesh.elem(:,3),:) ;
  PermK = PermK + GAUSSQUAD.TRIG(k,4) * fxnK(qp);
end

return;