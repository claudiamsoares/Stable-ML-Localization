function [estimate, varargout] = StableML_Localization(init, anchors, ...
                                                  distances, ...
                                                  varargin)
  % STABLEML_LOCALIZATION Implements Maximum-Likelihood sensor
  % network localization from noisy range measurements and a small
  % amount of anchor positions. For more information see <a href="matlab: 
  % web('http://bit.ly/SimpleStable14')">C. Soares, J. Xavier and
  % J. Gomes, “Distributed, simple and stable network
  % localization,” in Signal and Information Processing
  % (GlobalSIP), 2014 IEEE Global Conference on, pp. 764-768, Dec
  % 2014. </a>.
  %
  % sensors = [2 -0.8; 3 -0.8; 1 0; 4 0.6; 2.5 0.8]';
  % anchors = [0, 1; 0, -1; 5, 1; 5, -1]';
  % distances = dist([anchors sensors]);
  % distances(distances > 2.2) = 0;
  % init = sensors + 0.01*randn(size(sensors));
  % [estimate, nIter] = StableML_Localization(init, anchors, distances);



nrOfAnchors = size(anchors,2);
dim = size(anchors,1);
nrOfElements = size(distances,1);
nrOfSensors = nrOfElements - nrOfAnchors;



[MAXITER, epsilon, stopWhenDone, savedata] = decodeVarargin(varargin);


 %% Protection code
if max(max(abs(distances-distances'))) > 1e-9 || min(distances(:)) ...
      < 0
  error('distance matrix is not correct.')
end

  %% end of protection code




adj = distances>0;
nd = sum(adj(nrOfAnchors+1:end,nrOfAnchors+1:end),2);
dmax = max(nd);
  %% max |A_i|
maxAi = max(sum(adj(1:nrOfAnchors,nrOfAnchors+1:end),1));

L = dmax + maxAi + 2;

  %arc-node incidence matrix
A = triu(distances(nrOfAnchors+1:end,nrOfAnchors+1:end));
nrOfEdges = nnz(A);
[edgeset(:,1), edgeset(:,2)] = find(A);
sones = ones(nrOfEdges,1);
IXedges = (1:nrOfEdges)';
C = kron(sparse([IXedges; IXedges],...
                edgeset(:),...
                [sones; -sones],...
                nrOfEdges,nrOfSensors),eye(dim));


K = dim*sum(sum(adj(1:nrOfAnchors,nrOfAnchors+1:end),1));
[aedgeset(:,1), aedgeset(:,2)] = find(adj(1:nrOfAnchors,nrOfAnchors+1:end));
eones = ones(K,1);
Xidxs = (1:K)';
Yidxs = [];
alph = [];
Ai = cell(nrOfSensors,1);
for n=1:nrOfSensors
  Ai{n} = aedgeset((aedgeset(:,2) == n),1);
  a = anchors(:,Ai{n});
  alph = [alph; a(:)];
  for m=1:numel(Ai{n})
    Yidxs = [Yidxs; ((n-1)*dim+1:n*dim)'];
  end
end

E = sparse(Xidxs,Yidxs,eones,K,nrOfSensors*dim);

B = C'*C + E'*E;

dvec = distances(sub2ind(size(distances),edgeset(:,1)+nrOfAnchors, ...
                         edgeset(:,2)+nrOfAnchors));
rvec = distances(sub2ind(size(distances),aedgeset(:,1), ...
                              aedgeset(:,2)+nrOfAnchors));

stdev = 0;
channel = @(a) a + stdev*randn(dim*nrOfSensors,1);

X = init(:);
Y = zeros(dim*nrOfEdges,1);


for n=1:nrOfEdges
  Y((n-1)*dim+1:n*dim) = init(:,edgeset(n,1)) - init(:,edgeset(n,2));
  Y((n-1)*dim+1:n*dim) = Y((n-1)*dim+1:n*dim)/norm(Y((n-1)*dim+1:n*dim)) ...
      * distances(edgeset(n,1)+nrOfAnchors,edgeset(n,2)+nrOfAnchors);
end



W = zeros(K,1);
for n=1:(K/dim)
  W((n-1)*dim+1:n*dim) = init(:,aedgeset(n,2)) - anchors(:,aedgeset(n,1));
  W((n-1)*dim+1:n*dim) = W((n-1)*dim+1:n*dim)/norm(W((n-1)*dim+1:n*dim)) ...
      * distances(aedgeset(n,1),aedgeset(n,2)+nrOfAnchors);
end

C = C/L;
B = B/L;
E = E/L;
Ealph = E'*alph;
alphL = alph/L;

In = eye(nrOfSensors*dim);


for iter = 1:MAXITER
  oldX = X;

  X = channel((In - B)*X + C'*Y + E'*W + Ealph);
  
  Y = proj((L-1)/L*Y + C*oldX,dvec);
  
  W = proj((L-1)/L*W + E*oldX - alphL, rvec);

  if savedata.p
    [savedata] = saveIntermediateSols(iter, X, oldX, savedata);
  end
  
  if stopWhenDone && (L*norm(B*X - C'*Y - E'*W - Ealph) < epsilon)
    break
  end;
     
end

estimate = reshape(X, dim, nrOfSensors);

varargout{1} = iter;

if savedata.p
  varargout{2} = savedata;
end

function v = proj(v,r)
dim = numel(v)/numel(r);

for k=1:numel(r);
  v((k-1)*dim+1:k*dim) = v((k-1)*dim+1:k*dim)/norm(v((k-1)*dim+1:k*dim)) ...
      *r(k);
end
