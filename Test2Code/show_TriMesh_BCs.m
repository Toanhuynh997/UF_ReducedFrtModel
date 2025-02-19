function [] = show_TriMesh_BCs(TriMesh,fwn, FractureEgs, DirichletEgs, k)


%% Mesh info 
% NumNds = TriMesh.NumNds;
% NumEms = TriMesh.NumEms;
node = TriMesh.node;
elem = TriMesh.elem;
edge = TriMesh.edge;
% if (ch1==1) 
%   NumEgs = TriMesh.NumEgs;
%   edge = TriMesh.edge;
% end
m = size(FractureEgs, 1);
%% Preparing: Geometry with mesh
xx = [node(elem(:,1),1)'; node(elem(:,2),1)'; node(elem(:,3),1)'];
yy = [node(elem(:,1),2)'; node(elem(:,2),2)'; node(elem(:,3),2)'];

xx_frt = [node(edge(FractureEgs(:), 1), 1)', node(edge(FractureEgs(:), 2), 1)'];
yy_frt = [node(edge(FractureEgs(:), 1), 2)', node(edge(FractureEgs(:), 2), 2)'];

xx_top_frt = node(edge(FractureEgs(m), 2), 1);
yy_top_frt = node(edge(FractureEgs(m), 2), 2);

xx_bot_frt = node(edge(FractureEgs(1), 1), 1);
yy_bot_frt = node(edge(FractureEgs(1), 1), 2);

% xx_Dirichlet = [node(edge(DirichletEgs(1), 1), 1), node(edge(DirichletEgs(1), 2), 1)];
% yy_Dirichlet = [node(edge(DirichletEgs(1), 1), 2), node(edge(DirichletEgs(1), 2), 2)];
%% Plotting 
figure(fwn);
fill(xx,yy,'w');
set(gca,'fontsize',18);
hold on 
fill(xx_frt, yy_frt, 'r', 'EdgeColor', 'r');
% hold on
% fill(xx_Dirichlet, yy_Dirichlet, 'g', 'EdgeColor', 'g');
hold on
scatter(xx_top_frt, yy_top_frt, 'red', 'filled')
hold on
scatter(xx_bot_frt, yy_bot_frt, 'red', 'filled')
axis off;

%% Preparing: Geometry without mesh
% x1 = 0;
% x2 = 2; 
% y1 = 0; 
% y2 = 1;
% x = [x1, x2, x2, x1, x1];
% y = [y1, y1, y2, y2, y1];
% 
% BC_point = [1 m];
% 
% xx_BCs_frt = zeros(2, 1);
% 
% xx_BCs_frt(1) = node(edge(FractureEgs(BC_point(1)), 1), 1);
% xx_BCs_frt(2) = node(edge(FractureEgs(BC_point(2)), 2), 1);
% 
% yy_BCs_frt = zeros(2, 1);
% yy_BCs_frt(1) = node(edge(FractureEgs(BC_point(1)), 1), 2);
% yy_BCs_frt(2) = node(edge(FractureEgs(BC_point(2)), 2), 2);
% %% Plotting
% figure(fwn +1)
% if k ==1
%     fill(x, y, 'white');
% end
% % hold on
% fill(xx_Dirichlet, yy_Dirichlet, 'g', 'EdgeColor', 'g');
% hold on 
% fill(xx_BCs_frt, yy_BCs_frt, 'r', 'EdgeColor', 'r');
% hold on
% scatter(xx_top_frt, yy_top_frt, 'red', 'filled')
% hold on
% scatter(xx_bot_frt, yy_bot_frt, 'red', 'filled')
% axis off
return;