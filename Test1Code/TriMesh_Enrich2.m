function TriMesh = TriMesh_Enrich2(TriMesh,BndryDescMat)
%% (Enrich2) Enriching a triangular mesh to incorporate boundary edge info, 
% since not many mesh generators are good enough to automatically generate 
% boundary edge info.  
% 
% Here we adopt a 6*NumBndryPcs Boundary Descrition Matrix 
%   BndryDescMat (or BDM) 
% Assuming domains are polygonal (boundaries are line segments), there is 
% one column for each linear piece: 
%   Row 1,2: The starting-point x,y-coordinates 
%   Row 3,4: The ending-point x,y-coordinates 
%   Row 5,6: The outward unit normal vector x,y-componets 
% The BDM is problem-dependent but the code is not.
% 
% James Liu, ColoState; 2012/07--2016/09

%% Mesh info 
NumEgs = TriMesh.NumEgs;
TriMesh.BndryEdge = zeros(NumEgs,1);

%%
for ig=1:NumEgs 
  % Utilizing the info obtained from TriMesh_Enrich1.m 
  % TriMesh.edge2elem(ig,2)>0 if it is an interior edge 
  if TriMesh.edge2elem(ig,2)>0 
    continue;
  end
  % Assuming now TriMesh.edge2elem(ig,2)==0 
  x1 = TriMesh.node(TriMesh.edge(ig,1),1);
  x2 = TriMesh.node(TriMesh.edge(ig,2),1);
  y1 = TriMesh.node(TriMesh.edge(ig,1),2);
  y2 = TriMesh.node(TriMesh.edge(ig,2),2);
  % Screening 
  for k=1:size(BndryDescMat,1)
        X1 = BndryDescMat(k,1);  Y1 = BndryDescMat(k,2);  % Starting point 
        X2 = BndryDescMat(k,3);  Y2 = BndryDescMat(k,4);  % Ending point 
        % disp((x1-X1)*(Y2-Y1)-(X2-X1)*(y1-Y1))
        % disp((x2-X1)*(Y2-Y1)-(X2-X1)*(y2-Y1))
        if (x1-X1)*(Y2-Y1)==(X2-X1)*(y1-Y1) && (x2-X1)*(Y2-Y1)==(X2-X1)*(y2-Y1)
          TriMesh.BndryEdge(ig) = k;
          break;
        end
  end
end

return;