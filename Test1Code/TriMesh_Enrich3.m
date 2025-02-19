function TriMesh = TriMesh_Enrich3(TriMesh,BndryDescMat)
%% (Enrich3) Enriching a triangular mesh with tertiary info such as 
%   edge unit normal/tangential vectors, WhichEdge, etc. 
% Assuming Enrich2 already excuted (flag=2), BndryDescMat needed 
% James Liu, ColoState; 2012/07--2016/09

%% Mesh info 
NumEms = TriMesh.NumEms;
NumEgs = TriMesh.NumEgs;
% Asssuming TriMesh.flag=2
LenEg = TriMesh.LenEg;

%% Tertiary mesh info: Element centers
k1 = TriMesh.elem(:,1);  k2 = TriMesh.elem(:,2);  k3 = TriMesh.elem(:,3);
x1 = TriMesh.node(k1,1);  y1 = TriMesh.node(k1,2); 
x2 = TriMesh.node(k2,1);  y2 = TriMesh.node(k2,2);
x3 = TriMesh.node(k3,1);  y3 = TriMesh.node(k3,2);
xc = (1.0/3)*(x1+x2+x3);
yc = (1.0/3)*(y1+y2+y3);
EmCntr = [xc, yc];

%% Tertiary mesh info: Edge normal/tangential vectors
AllEg = TriMesh.node(TriMesh.edge(:,2),:) ...
      - TriMesh.node(TriMesh.edge(:,1),:);
LenEg = sqrt(AllEg(:,1).^2+AllEg(:,2).^2);
TanEg = AllEg./[LenEg,LenEg];
NmlEg = [TanEg(:,2),-TanEg(:,1)];  % Rotating clockwise 90 degree
% JL20130801: TO BE REVISED FOR EFFICIENCY 
% Correcting normal vectors 
% for ig=1:NumEgs
%   if TriMesh.BndryEdge(ig)>0 
%     NmlEg(ig,:) = BndryDescMat(TriMesh.BndryEdge(ig),5:6);
%     TanEg(ig,:) = [NmlEg(ig,2),-NmlEg(ig,1)];
%   end
% end
ig = find(TriMesh.BndryEdge>0);
NmlEg(ig,:) = BndryDescMat(TriMesh.BndryEdge(ig),5:6);
TanEg(ig,:) = [NmlEg(ig,2),-NmlEg(ig,1)];

% %% Edge midpoints
% EgMidPt = 0.5*(TriMesh.node(TriMesh.edge(:,1),:) ...
%              + TriMesh.node(TriMesh.edge(:,2),:));

%% Which edge
AuxMat = sparse(NumEms,NumEgs);
for j=1:3
  AuxMat = AuxMat + sparse((1:NumEms),TriMesh.elem2edge(:,j),...
    j*ones(1,NumEms),NumEms,NumEgs);
end
WhichEdge = zeros(NumEgs,2);
for ig=1:NumEgs
  WhichEdge(ig,1) = AuxMat(TriMesh.edge2elem(ig,1),ig);
  if TriMesh.edge2elem(ig,2)>0
    WhichEdge(ig,2) = AuxMat(TriMesh.edge2elem(ig,2),ig);
  end
end

%% For each element, three edge midpoints
xm = zeros(NumEms,3);  ym = zeros(NumEms,3);
xm(:,1) = 0.5*(x2+x3);  ym(:,1) = 0.5*(y2+y3);
xm(:,2) = 0.5*(x3+x1);  ym(:,2) = 0.5*(y3+y1);
xm(:,3) = 0.5*(x1+x2);  ym(:,3) = 0.5*(y1+y2);
% Signs for elementwise edge normals 
DirVec1 = [xm(:,1),ym(:,1)] - [xc,yc];
DirVec2 = [xm(:,2),ym(:,2)] - [xc,yc];
DirVec3 = [xm(:,3),ym(:,3)] - [xc,yc];
% Signs 
sn = zeros(NumEms,3);
sn(:,1) = sign(dot(DirVec1,NmlEg(TriMesh.elem2edge(:,1),:),2));
sn(:,2) = sign(dot(DirVec2,NmlEg(TriMesh.elem2edge(:,2),:),2));
sn(:,3) = sign(dot(DirVec3,NmlEg(TriMesh.elem2edge(:,3),:),2));
% Further signs, TO BE REVISED FOR EFFICIENCY 
AuxMat = sparse(NumEms,NumEgs);
for j=1:3
  AuxMat = AuxMat + sparse((1:NumEms),TriMesh.elem2edge(:,j),...
    sn(:,j),NumEms,NumEgs);
end
sns = zeros(NumEgs,2);
for ig=1:NumEgs
  sns(ig,1) = AuxMat(TriMesh.edge2elem(ig,1),ig);
  if TriMesh.edge2elem(ig,2)>0
    sns(ig,2) = AuxMat(TriMesh.edge2elem(ig,2),ig);
  end
end

%% Finishing: flag=3 for tertiary info 
TriMesh.EmCntr = EmCntr;
% TriMesh.EgMidPt = EgMidPt;
TriMesh.NmlEg = NmlEg;
TriMesh.TanEg = TanEg;
TriMesh.WhichEdge = WhichEdge;
TriMesh.SignEmEg = sn;
TriMesh.SignEgEm = sns;
TriMesh.flag = 3;

return;