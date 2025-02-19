function [] = show_TriMesh_ScaVecEm_mix(TriMesh, ScaEm,VecEm, fwn, ts, mf)
%% TriMesh: Show scalar+vector (emws.const.) as a mix of color image and quiver
% fwn: figure window number 
%  ts: title string for the figure window
%  mf: magnifying factor for vectors (length magnified mf times)  
% James Liu, ColoState; 2012/07--2017/02

%% Mesh info 
node = TriMesh.node;
elem = TriMesh.elem;

%% Preparing  
xx = [node(elem(:,1),1)'; node(elem(:,2),1)'; node(elem(:,3),1)'];
yy = [node(elem(:,1),2)'; node(elem(:,2),2)'; node(elem(:,3),2)'];
sca = [ScaEm'; ScaEm'; ScaEm'];
% % % % % % % 
% % % % % % % %% Plotting the elementwise pressure 
figure(fwn);  % figure window number 
set(gca,'fontsize',18);
H1 = patch(xx,yy,sca); 
set(H1,'edgecolor','interp'); 
colormap jet;
colorbar('location','eastoutside','fontsize',14);
axis square;  
% axis equal;  
axis tight;
hold on;
figure(fwn+1)
quiver(TriMesh.EmCntr(:,1),TriMesh.EmCntr(:,2),...
  VecEm(:,1),VecEm(:,2), mf ,'b');  % vectors magnified mf times 
% for the figure 
title(ts);  % title string 
axis square;  
% axis equal;  
axis tight;

return;