clear;
load('TestUnitedSF_RefSol_PureDiff_v1.mat', 'ExactState')
load('UnitedFilter_2DRft_3Para_100LinearPartialOb_v1.mat',...
    'Est_State', 'rmse', 'nx', 'ny', 'nttrue', 'ntEnSF', 'ApprPara')
% load('UnitedFilter_2DRft_3Para_75LinearPartialOb_v1.mat',...
%     'Est_State', 'rmse', 'nx', 'ny', 'nttrue', 'ntEnSF', 'ApprPara')
% load('UnitedFilter_2DRft_3Para_50PartialOb_Mixed_Redo_Noise0001_v1.mat',...
%     'Est_State', 'rmse', 'nx', 'ny', 'nttrue', 'ntEnSF', 'ApprPara')

TrueState = double(ExactState');
FilterState = double(Est_State');

format compact;  format shortG; 

Para = double(ApprPara);
Para1 = Para(:, 1);
Para2 = Para(:, 2);
Para3 = Para(:, 3);

xt = 0;
xe = 2;
yt = 0;
ye = 1;
nx = double(nx);
ny = double(ny); 
node_frt = ny+1;

nttrue = double(nttrue);
ntEnSF = double(ntEnSF);
T = 1;
t0 =0;

h = (ye-yt)/ny;

frt_length = 1e-03;
diff_frt = 2/frt_length;

poros_subdom = 1;
poros_gamma = 1/frt_length;
poros_frt = frt_length*poros_gamma;

alpha_frt = diff_frt*frt_length;

TriMesh = MeshGenerator_PureDiff_v2(xt, xe, yt, ye, nx, ny);

diffusion = cell(2, 1);

for k=1:2
    diffusion{k} = zeros(TriMesh{k}.NumEms, 1);
end

diff_1subdom = 1;
diff_2subdom = 1;

NumVert = cell(2, 1);

NumVert{1} = 1;
NumVert{2} = 1;

DOFs_Local = cell(2, 1);
FractureEgs   = cell(2, 1);
edge = cell(2, 1);
elem = cell(2, 1);
elem2edge = cell(2, 1);
node = cell(2, 1);
for k=1:2
    FractureEgs{k} = TriMesh{k}.FractureEgs;
    DOFs_Local{k} = TriMesh{k}.DOFs;
    edge{k} = TriMesh{k}.edge;
    elem{k} = TriMesh{k}.elem;
    elem2edge{k} = TriMesh{k}.elem2edge;
    node{k} = TriMesh{k}.node;
end
GAUSSQUAD = SetGaussQuad(2,4,3);

time_plotEnSF = ntEnSF+1;
time_plottrue = nttrue+1;

DOFs = DOFs_Local{1}+DOFs_Local{2};

TrueSln = cell(2, 1);
TrueSln{1} = TrueState(1:DOFs_Local{1}, time_plottrue);
TrueSln{2} = TrueState(DOFs_Local{1} +(1:DOFs_Local{2}), time_plottrue);

FilterSln = cell(2, 1);
FilterSln{1} = FilterState(1:DOFs_Local{1}, time_plotEnSF);
FilterSln{2} = FilterState(DOFs_Local{1} +(1:DOFs_Local{2}), time_plotEnSF);

NumerPresEm = cell(2, 1);
NumerVelEmCntr = cell(2, 1);
[NumerPresEm{1}, NumerVelEmCntr{1}, NumerFlux{1}, ~, ~] = ...
  Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{1}.BndryDescMat, TriMesh{1}, ...
  TrueSln{1});
% 
show_TriMesh_ScaVecEm_mix(0, 1, nx/2, ny, ...
    TriMesh{1}, NumerPresEm{1}, NumerVelEmCntr{1}, ...
  23, '', 1.5);
% 
hold on 
[NumerPresEm{2}, NumerVelEmCntr{2}, NumerFlux{2}, ~, ~] = ...
  Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{2}.BndryDescMat, TriMesh{2},...
  TrueSln{2});
% % 
show_TriMesh_ScaVecEm_mix(1, 2, nx/2, ny, ...
    TriMesh{2}, NumerPresEm{2}, NumerVelEmCntr{2}, ...
  23, '', 1.5);

hold on
last_truevel_frt = TrueState(DOFs_Local{1}+DOFs_Local{2}+(1:(ny+1)),...
    time_plottrue);
NumerVelEmCntr_frt = ...
    Vel_Darcyflux_frt(last_truevel_frt, size(FractureEgs{1}, 1));

xgrid_Z = linspace(1, 2, nx);
ygrid_Z = linspace(0, 1, 2*ny);

PresAll = [NumerPresEm{1}; NumerPresEm{2}];
Zq = reshape(PresAll, 2*ny, nx);

[XZ, YZ] = meshgrid(xgrid_Z, ygrid_Z);
figure(50)
contourf(XZ, YZ, Zq)
axis square
axis tight

yy_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 2);
xx_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 1);

figure(24)
quiver(xx_frt, yy_frt, zeros(size(FractureEgs{1}(:), 1), 1),...
    0.01*NumerVelEmCntr_frt, 'r', 'AutoScale', 'off');

NumerPresEm_Filter = cell(2, 1);
NumerVelEmCntr_Filter = cell(2, 1);
[NumerPresEm_Filter{1}, NumerVelEmCntr_Filter{1}, NumerFlux{1}, ~, ~] = ...
  Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{1}.BndryDescMat, TriMesh{1}, ...
  FilterSln{1});
% 
show_TriMesh_ScaVecEm_mix(0, 1, nx/2, ny, ...
    TriMesh{1}, NumerPresEm_Filter{1},...
    NumerVelEmCntr_Filter{1},27, '', 1.5);
% 
hold on 
[NumerPresEm_Filter{2}, NumerVelEmCntr_Filter{2}, NumerFlux{2}, ~, ~] = ...
  Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{2}.BndryDescMat, TriMesh{2},...
  FilterSln{2});
% % 
show_TriMesh_ScaVecEm_mix(1, 2, nx/2, ny, TriMesh{2}, NumerPresEm_Filter{2}, ...
    NumerVelEmCntr_Filter{2}, 27, '', 1.5);

hold on
last_filtervel_frt = FilterState(DOFs_Local{1}+DOFs_Local{2}+(1:(ny+1)),...
    time_plotEnSF);
NumerVelEmCntr_frt =...
    Vel_Darcyflux_frt(last_filtervel_frt, size(FractureEgs{1}, 1));


PresAll_Filter = [NumerPresEm_Filter{1}; NumerPresEm_Filter{2}];
Zq_Filter = reshape(PresAll_Filter, 2*ny, nx);

figure(60)
contourf(XZ, YZ, Zq_Filter)
axis square
axis tight

yy_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 2);
xx_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 1);

figure(28)
quiver(xx_frt, yy_frt, zeros(size(FractureEgs{1}(:), 1), 1),...
    0.01*NumerVelEmCntr_frt, 'r', 'AutoScale', 'off');

last_truepres_frt = TrueState(DOFs_Local{1}+DOFs_Local{2}+ny+1+(1:ny),...
    time_plottrue);
last_filterpres_frt = FilterState(DOFs_Local{1}+DOFs_Local{2}+ny+1+(1:ny),...
    time_plotEnSF);

ygrid = linspace(0, 1, ny+1);
ymid = (ygrid(2:end)+ygrid(1:end-1))/2;
figure(30)
plot(ymid , last_truepres_frt, '-b')
hold on 
plot(ymid , last_filterpres_frt, '-r')
legend('True $$p_{\gamma}$$', 'Filter $$\tilde{p}_{\gamma}$$')


figure(31)
plot(ygrid, last_truevel_frt, '-b')
hold on 
plot(ygrid, last_filtervel_frt, '-r')
legend('True $$u_{\gamma}$$', 'Filter $$\tilde{u}_{\gamma}$$')

figure(32)
plot(Para1, '-o', 'Color', 'b')
hold on
plot(ones(size(Para1, 1)), 'Color', 'r')
legend('Approximate $$k_1$$', 'True $$k_1$$')
ylim([0, 10])
figure(33)
plot(Para2, '-o', 'Color', 'b')
hold on
plot(ones(size(Para2, 1)), 'Color', 'r')
legend('Approximate $$k_2$$', 'True $$k_2$$')
ylim([0, 10])

figure(34)
plot(Para3, '-o', 'Color', 'b')
hold on
plot(2*ones(size(Para3, 1)), 'Color', 'r')
legend('Approximate $$\alpha\_frt$$', 'True $$\alpha\_frt$$')