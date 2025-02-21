clear;

load('UnitedSF_RefSol_AdvDiff_v1.mat', 'nx', 'ny', 'ExactState')

load('UnitedFilter_2DRft_AdvDiff_50MixedPartialOb_Noise0001_ReScale_v1.mat',...
    'Est_State', 'rmse', 'ApprPara', 'nttrue', 'ntEnSF')
diff1 = 3.15e-04;
diff2 = 3.15e-04;
alp_frt = 9.92e-04;

Para = double(ApprPara);

Para1 = Para(:, 1);
Para2 = Para(:, 2);
Para3 = Para(:, 3);

% nx = double(nx);
% ny = double(ny);

NPX = 2;
NPY = 1;
nt = 50;
xa = 0;
xb = 2;
yc = 0;
yd = 1;
hy = (yd-yc)/ny;
% ref_hy = (yd-yc)/ref_ny;
nodal_points = linspace(yc,yd, ny+1);
ispre = 1;

gnx=nx*NPX; 
gny=ny*NPY;
xgrid_all = linspace(xa,xb,gnx+1);
ygrid_all = linspace(yc,yd, gny+1);

xmid_all = (xgrid_all(1:end-1)+xgrid_all(2:end))/2;
ymid_all = (ygrid_all(1:end-1)+ygrid_all(2:end))/2;


[Xmid_all,Ymid_all]=meshgrid(xmid_all, ymid_all);

subdomain = CreateSubdom_Setting(nx, ny, NPX, NPY);
% [~, ~, subdomain]=...
%     main_mono_frac_transport_dis(nx, ny, NPX, NPY, nt);

RectMesh = cell(2, 1);
NumEms = cell(2, 1);
NumDofus = cell(2, 1);
n_ori_out = cell(2, 1);

for k = 1:2
    RectMesh{k} = subdomain{k}.RectMesh;
    NumEms{k} = RectMesh{k}.NumEms;
    NumDofus{k} = RectMesh{k}.NumDofus;
    n_ori_out{k}= subdomain{k}.n_ori_out;
end

Sol = double(ExactState');
EnSFSol = double(Est_State');

ExactSol_decomp = cell(3, 1);

ExactSol_decomp{1} = Sol(1:n_ori_out{1}, :);
ExactSol_decomp{2} = Sol(n_ori_out{1}  + (1:n_ori_out{2}), :);
ExactSol_decomp{3} = Sol(n_ori_out{1}+ n_ori_out{2} + ...
    (1:(2*ny +ny + ny+1)), :);

% ExactSol_decomp{1} = Sol(1:n_ori_out{1}, 1);
% ExactSol_decomp{2} = Sol(n_ori_out{1}  + (1:n_ori_out{2}), 1);
% ExactSol_decomp{3} = Sol(n_ori_out{1}+ n_ori_out{2} + ...
%     (1:(2*ny +ny + ny+1)), 1);

EnSFSol_decomp = cell(3, 1);

% EnSFSol_decomp{1} = EnSFSol(1:n_ori_out{1}, 1);
% EnSFSol_decomp{2} = EnSFSol(n_ori_out{1}  + (1:n_ori_out{2}), 1);
% EnSFSol_decomp{3} = EnSFSol(n_ori_out{1}+ n_ori_out{2} + ...
%     (1:(2*ny +ny + ny+1)), 1);

EnSFSol_decomp{1} = EnSFSol(1:n_ori_out{1}, :);
EnSFSol_decomp{2} = EnSFSol(n_ori_out{1}  + (1:n_ori_out{2}), :);
EnSFSol_decomp{3} = EnSFSol(n_ori_out{1}+ n_ori_out{2} + ...
    (1:(2*ny +ny + ny+1)), :);

uh_ref = cell(3, 1);
ph_ref = cell(3, 1);
the_ref = cell(3, 1);

for k = 1:2
    uh_ref{k} = ExactSol_decomp{k}(1:RectMesh{k}.NumDofus, :);
    ph_ref{k} = ExactSol_decomp{k}(RectMesh{k}.NumDofus+...
        (1:RectMesh{k}.NumEms), :); 
    the_ref{k} = ExactSol_decomp{k}((NumDofus{k}+NumEms{k}+1):end, :);
end
uh_ref{3} = ExactSol_decomp{3}(1:2*ny, :);
ph_ref{3} = ExactSol_decomp{3}(2*ny + (1:ny), :);
the_ref{3} = ExactSol_decomp{3}(3*ny+(1:ny+1), :);

uh_EnSF = cell(3, 1);
ph_EnSF = cell(3, 1);
the_EnSF = cell(3, 1);

for k = 1:2
    uh_EnSF{k} = EnSFSol_decomp{k}(1:RectMesh{k}.NumDofus, :);
    ph_EnSF{k} = EnSFSol_decomp{k}(RectMesh{k}.NumDofus+...
        (1:RectMesh{k}.NumEms), :);    
    the_EnSF{k} = EnSFSol_decomp{k}((NumDofus{k}+NumEms{k}+1):end, :);
end

uh_EnSF{3} = EnSFSol_decomp{3}(1:2*ny, :);
ph_EnSF{3} = EnSFSol_decomp{3}(2*ny + (1:ny), :);
the_EnSF{3} = EnSFSol_decomp{3}(3*ny+(1:ny+1), :);

TrueFlux = Vel_Frt_Center(uh_ref{3}(:, end), ny);
EnSFFlux = Vel_Frt_Center(uh_EnSF{3}(:, end), ny);

xfrt = ones(ny, 1);

ph_refPlot = cell(2, 1);
ph_EnSFPlot = cell(2, 1);

for k =1:2
    ph_refPlot{k} =(reshape(ph_ref{k}(:, end),nx,ny))';
    ph_EnSFPlot{k} =(reshape(ph_EnSF{k}(:, end),nx,ny))';
end
% 
phref_all =[ph_refPlot{1} ph_refPlot{2}];
% 
figure(1)
contourf(Xmid_all, Ymid_all, phref_all)

figure(2)
pcolor(Xmid_all, Ymid_all, phref_all)
colormap(jet)
shading interp

phEnSF_all =[ph_EnSFPlot{1} ph_EnSFPlot{2}];
% 
figure(3)
contourf(Xmid_all, Ymid_all, phEnSF_all)

figure(4)
pcolor(Xmid_all, Ymid_all, phEnSF_all)
colormap(jet)
shading interp


Vel_xmidref = cell(2, 1);
Vel_ymidref = cell(2, 1);

for k = 1:2
    [Vel_xmidref{k}, Vel_ymidref{k}] = ...
        Compute_Vel_Center(RectMesh{k}, ExactSol_decomp{k}(:, end));
end

for kk = 1:2
    Vel_xmidref{kk} = reshape(Vel_xmidref{kk}, nx, ny)';
    Vel_ymidref{kk} = reshape(Vel_ymidref{kk}, nx, ny)';
end

uhx_ref = [Vel_xmidref{1} Vel_xmidref{2}];
uhy_ref = [Vel_ymidref{1} Vel_ymidref{2}];


Vel_xmidEnSF = cell(2, 1);
Vel_ymidEnSF = cell(2, 1);

for k = 1:2
    [Vel_xmidEnSF{k}, Vel_ymidEnSF{k}] = ...
        Compute_Vel_Center(RectMesh{k}, EnSFSol_decomp{k}(:, end));
end

for kk = 1:2
    Vel_xmidEnSF{kk} = reshape(Vel_xmidEnSF{kk}, nx, ny)';
    Vel_ymidEnSF{kk} = reshape(Vel_ymidEnSF{kk}, nx, ny)';
end

uhx_EnSF = [Vel_xmidEnSF{1} Vel_xmidEnSF{2}];
uhy_EnSF = [Vel_ymidEnSF{1} Vel_ymidEnSF{2}];

figure(5)
quiver(Xmid_all, Ymid_all, uhx_ref, uhy_ref)
hold on
quiver(xfrt, ygrid_all(1:end-1), zeros(ny, 1), TrueFlux);

figure(6)
quiver(Xmid_all, Ymid_all, uhx_EnSF, uhy_EnSF)
hold on
quiver(xfrt, ygrid_all(1:end-1), zeros(ny, 1), EnSFFlux);

figure(7)
plot(ph_ref{3}(:, end))
hold on
plot(ph_EnSF{3}(:, end))

figure(8)
plot(uh_ref{3}(:, end))
hold on
plot(uh_EnSF{3}(:, end))

figure(9)
plot(TrueFlux)
hold on 
plot(EnSFFlux)
% figure(10)
% plot(Para1 / 10000, '-o', 'Color', 'b')
%     hold on
% plot(diff1*ones(size(Para1, 1)), 'Color', 'r')
%     ylim([0, 2.5e-03])
% 
% figure(11)
% plot(Para2 / 10000, '-o', 'Color', 'b')
% hold on
% plot(diff2*ones(size(Para2, 1)), 'Color', 'r')
% ylim([0, 2.5e-03])
% 
% figure(12)
% plot(Para3 / 10000, '-o', 'Color', 'b')
% hold on
% plot(alp_frt*ones(size(Para3, 1)), 'Color', 'r')
% ylim([0, 7e-03])

figure(10)
plot(1 ./ Para1, '-o', 'Color', 'b')
    hold on
plot(diff1*ones(size(Para1, 1)), 'Color', 'r')
    ylim([0, max(1 ./ Para1)+5e-04])

figure(11)
plot(1 ./ Para2, '-o', 'Color', 'b')
hold on
plot(diff2*ones(size(Para2, 1)), 'Color', 'r')
ylim([0, max(1 ./ Para2)+5e-04])

figure(12)
plot(1 ./ Para3, '-o', 'Color', 'b')
hold on
plot(alp_frt*ones(size(Para3, 1)), 'Color', 'r')
ylim([0, max(1 ./ Para3)+5e-04])