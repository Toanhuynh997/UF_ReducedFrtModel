function [NumerPresEm,NumerVelEmCntr,NumerFlux,FluxDscp,LMCR] = ...
  Darcy_MFEM_TriRT0P0_PresVelFlux(BndryDescMat, TriMesh, sln)
% Darcy: MFEM(RT0,P0): Computing (numerical) pressure, velocity, flux, etc. 
% James Liu, ColoState; 2012/07--2018/12 

%% Mesh info 
NumEms = TriMesh.NumEms;
NumEgs = TriMesh.NumEgs;
LenEg = TriMesh.LenEg;
e2g = TriMesh.elem2edge;
sn = TriMesh.SignEmEg;
area = TriMesh.area;

%% "Computing" numerical pressure: Elementwise constant (cell averages) 
NumerPresEm = sln(NumEgs+(1:NumEms));

%% Computing numerical velocity at element centers 
CofRT0 = zeros(NumEms,3);
for j=1:3
  CofRT0(:,j) = sln(TriMesh.elem2edge(:,j)).*TriMesh.SignEmEg(:,j);
end
BasFxn = zeros(NumEms,3,2);  % All elts., 3 edges, 2-vec. 
cntr = (1/3)*(TriMesh.node(TriMesh.elem(:,1),:)...
            + TriMesh.node(TriMesh.elem(:,2),:)...
            + TriMesh.node(TriMesh.elem(:,3),:));
for j=1:3
  %% JL20160614: THERE IS AN ISSUE ABOUT NEGATIVE SIGN !!!
  coeff = -LenEg(TriMesh.elem2edge(:,j))./(2*area);
  BasFxn(:,j,:) = [coeff,coeff].*(cntr-TriMesh.node(TriMesh.elem(:,j),:));
end
% NumerVelEmCntr = zeros(NumEms,2);
NumerVelEmCntr = [CofRT0(:,1),CofRT0(:,1)] .* squeeze(BasFxn(:,1,:))...
               + [CofRT0(:,2),CofRT0(:,2)] .* squeeze(BasFxn(:,2,:))...
               + [CofRT0(:,3),CofRT0(:,3)] .* squeeze(BasFxn(:,3,:));
NumerVelEmCntr = -NumerVelEmCntr;

%% Computing numerical flux: For all elements, 3 edges 
NumerFlux = zeros(NumEms,3);
coeff = sln(1:NumEgs);
for j=1:3
  NumerFlux(:,j) = coeff(e2g(:,j)) .* sn(:,j) .* LenEg(e2g(:,j));
end

%% Computing elementwise local mass-conservation residuals: 0 theoretically 
LMCR = zeros(TriMesh.NumEms,1);

%% Computing flux discrepancy aross edges: 0 theoretically  
FluxDscp = zeros(TriMesh.NumEgs,1);

return;