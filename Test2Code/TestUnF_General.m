clear;
load('TestEnSF_RefSol_General_Aniso_v1.mat', 'ExactState')
Glob_Sln = double(ExactState');

% load('UnitedFilter_2DRft_50PartialOb_Mixed_Aniso_Noise0001_v1.mat',...
%     'Est_State', 'rmse', 'nx', 'ny', 'nttrue', 'ntEnSF', 'ApprPara')
% load('UnitedFilter_2DRft_50PartialOb_Mixed_Aniso_Noise001_v1.mat',...
%     'Est_State', 'rmse', 'nx', 'ny', 'nttrue', 'ntEnSF', 'ApprPara')
% load('UnitedFilter_2DRft_50PartialOb_Mixed_Aniso_Noise01_v1.mat',...
%     'Est_State', 'rmse', 'nx', 'ny', 'nttrue', 'ntEnSF', 'ApprPara')

Appr_Sln = double(Est_State');

format compact;  format shortG; 
xt = 0;
xe = 2;
yt = 0;
ye = 1;
nx = double(nx);
ny = double(ny);

node_frt = ny +1;

nttrue = double(nttrue);
ntEnSF = double(ntEnSF);

Para = double(ApprPara);

rmse = double(rmse);

Para1 = Para(:, 1);
Para2 = Para(:, 2);
Para3 = 500*Para(:, 3);
% Para4 = Para(:, 4);

T = 1;
t0 =0;

h = (ye-yt)/ny;

% dt = (T-t0)/nt;
% timegrid = linspace(0, T, nt+1);
% diff_frt = 1e+3;
frt_length = 1e-03;

pres_bot_frt = 1;
pres_top_frt = 0;

poros_subdom = 1;
poros_frt = 1;

TriMesh = MeshGenerator_PureDiff_General(xt, xe, yt, ye, nx, ny);

DOFs_Local = cell(2, 1);
FractureEgs   = cell(2, 1);
edge = cell(2, 1);
node = cell(2, 1);
for k=1:2
    FractureEgs{k} = TriMesh{k}.FractureEgs;
    DOFs_Local{k} = TriMesh{k}.DOFs;
    edge{k} = TriMesh{k}.edge;
    node{k} = TriMesh{k}.node;
end
GAUSSQUAD = SetGaussQuad(2,4,3);

% toc
ApprSln = cell(3, 1);
RefSln = cell(3, 1);

ApprSln{1} = Appr_Sln(1:DOFs_Local{1}, :);
ApprSln{2} = Appr_Sln(DOFs_Local{1} +(1:DOFs_Local{2}), :);
ApprSln{3} = Appr_Sln(DOFs_Local{1}+DOFs_Local{2}+(1:(2*ny+1)), :);

RefSln{1} = Glob_Sln(1:DOFs_Local{1}, :);
RefSln{2} = Glob_Sln(DOFs_Local{1} +(1:DOFs_Local{2}), :);
RefSln{3} = Glob_Sln(DOFs_Local{1}+DOFs_Local{2}+(1:(2*ny+1)), :);

%% Reference
NumerPresEm = cell(2, 1);
NumerVelEmCntr = cell(2, 1);
[NumerPresEm{1}, NumerVelEmCntr{1}, NumerFlux{1}, ~, ~] = ...
    Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{1}.BndryDescMat, TriMesh{1}, ...
    RefSln{1}(:, end));
% 
show_TriMesh_ScaVecEm_mix(TriMesh{1}, NumerPresEm{1}, NumerVelEmCntr{1}, ...
    12, '', 1);
% 
hold on 
[NumerPresEm{2}, NumerVelEmCntr{2}, NumerFlux{2}, ~, ~] = ...
    Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{2}.BndryDescMat, TriMesh{2},...
    RefSln{2}(:, end));
% % 
show_TriMesh_ScaVecEm_mix(TriMesh{2}, NumerPresEm{2}, NumerVelEmCntr{2}, ...
    12, '', 1);

hold on
last_vel_frt = Glob_Sln(DOFs_Local{1}+DOFs_Local{2}+(1:(ny+1)),...
    end);
NumerVelEmCntr_frt = ...
    Vel_Darcyflux_frt(last_vel_frt, size(FractureEgs{1}, 1));

yy_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 2);
xx_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 1);

figure(13)
quiver(xx_frt, yy_frt, zeros(size(FractureEgs{1}(:), 1), 1),...
    0.1*NumerVelEmCntr_frt, 'r', 'AutoScale', 'off');

xgrid_Z = linspace(1, 2, nx);
ygrid_Z = linspace(0, 1, 2*ny);

[XZ, YZ] = meshgrid(xgrid_Z, ygrid_Z);

PresAll = [NumerPresEm{1}; NumerPresEm{2}];
Zq = reshape(PresAll, 2*ny, nx);

figure(31)
contourf(XZ, YZ, Zq)
colormap jet;
colorbar('location','eastoutside','fontsize',14);
axis square;  
% axis equal;  
axis tight;

figure(32)
pcolor(XZ, YZ, Zq)
shading interp
colormap jet
axis square;  
% axis equal;  
axis tight;

%% Approximation
NumerPresEm_Filter = cell(2, 1);
NumerVelEmCntr_Filter = cell(2, 1);
[NumerPresEm_Filter{1}, NumerVelEmCntr_Filter{1}, NumerFlux{1}, ~, ~] = ...
    Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{1}.BndryDescMat, TriMesh{1}, ...
    ApprSln{1}(:, end));
% 
show_TriMesh_ScaVecEm_mix(TriMesh{1}, NumerPresEm_Filter{1}, NumerVelEmCntr_Filter{1}, ...
    23, '', 1);
% 
hold on 
[NumerPresEm_Filter{2}, NumerVelEmCntr_Filter{2}, NumerFlux{2}, ~, ~] = ...
    Darcy_MFEM_TriRT0P0_PresVelFlux(TriMesh{2}.BndryDescMat, TriMesh{2},...
    ApprSln{2}(:, end));
% % 
show_TriMesh_ScaVecEm_mix(TriMesh{2}, NumerPresEm_Filter{2}, NumerVelEmCntr_Filter{2}, ...
    23, '', 1);

hold on
last_vel_frt_filter = Appr_Sln(DOFs_Local{1}+DOFs_Local{2}+(1:(ny+1)),...
    end);
NumerVelEmCntr_frt_filter = ...
    Vel_Darcyflux_frt(last_vel_frt_filter, size(FractureEgs{1}, 1));

yy_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 2);
xx_frt = node{1}(edge{1}(FractureEgs{1}(:), 1), 1);

figure(24)
quiver(xx_frt, yy_frt, zeros(size(FractureEgs{1}(:), 1), 1),...
    0.1*NumerVelEmCntr_frt_filter, 'r', 'AutoScale', 'off');

xgrid_Z = linspace(1, 2, nx);
ygrid_Z = linspace(0, 1, 2*ny);

[XZ, YZ] = meshgrid(xgrid_Z, ygrid_Z);

PresAll_Filter = [NumerPresEm_Filter{1}; NumerPresEm_Filter{2}];
Zq_Filter = reshape(PresAll_Filter, 2*ny, nx);

figure(51)
contourf(XZ, YZ, Zq_Filter)
colormap jet;
colorbar('location','eastoutside','fontsize',14);
axis square;  
% axis equal;  
axis tight;

figure(52)
pcolor(XZ, YZ, Zq_Filter)
colormap jet
shading interp
axis square;  
% axis equal;  
axis tight;

%% Parameters
figure(1)
plot(Para1, '-o', 'Color', 'b')
hold on
plot(ones(size(Para1, 1)), 'Color', 'r')
legend('Approximate $$k_1$$', 'True $$k_1$$')
ylim([0, max(Para1)+0.5])
figure(2)
plot(Para2, '-o', 'Color', 'b')
hold on
plot(ones(size(Para2, 1)), 'Color', 'r')
legend('Approximate $$d_{\gamma, 1}$$', 'True $$d_{\gamma, 1}$$')
ylim([0, max(Para2)+0.5])

figure(3)
plot(Para3, '-o', 'Color', 'b')
hold on
plot(2000*ones(size(Para3, 1)), 'Color', 'r')
legend('Approximate $$d_{\gamma, 2}$$', 'True $$d_{\gamma,2}$$')
ylim([min(Para3)-100, max(Para3)+100])


%% 1D
figure(5)
plot(RefSln{3}(1:(ny+1), end))
hold on
plot(ApprSln{3}(1:(ny+1), end))

figure(6)
plot(RefSln{3}(ny+1+(1:ny), end))
hold on
plot(ApprSln{3}(ny+1+(1:ny), end))

figure(7)
plot(rmse)