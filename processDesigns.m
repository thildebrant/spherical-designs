function data = processDesigns(folder, tol, doExact)
% PROCESSDESIGNS  Batch loader & checker for Sloane spherical designs
%
%   data = processDesigns(folder, tol, doExact)
%
%   folder   – directory containing   des*.txt         (default = pwd)
%   tol      – duplicate / rat() tolerance             (default = 1e-12)
%   doExact  – true  → convert coordinates to sym      (default = false)
%
%   Each data(k) record contains:
%       .file        - filename
%       .ptsFloat    - Nx3 double   (original)
%       .ptsSym      - Nx3 sym      (optional, empty when doExact = false)
%       .faces       - Fx3  int32   (convex-hull triangulation, empty if none)
%       .q           - struct with quality metrics:
%                        eulerDefect, AR_max, minEdge, maxEdge,
%                        minAngle,  maxMomentErr  (t-strength residual)

    if nargin < 1, folder  = pwd;      end
    if nargin < 2, tol     = 1e-12;    end
    if nargin < 3, doExact = false;    end

    files  = dir(fullfile(folder,'des*.txt'));
    nFiles = numel(files);
    if nFiles == 0
        error('No des*.txt files found in "%s".', folder);
    end
    fprintf('▶  Processing %d design files  (exact = %d)\n', nFiles, doExact);

    data = struct([]);
    for k = 1:nFiles
        %-------------------------------------------------------------- load
        fname = fullfile(files(k).folder, files(k).name);
        C     = readmatrix(fname);              % Nx3 double
        if size(C,2) ~= 3
            warning('File %s is not 3-column—skipped.', files(k).name);
            continue
        end

        %----------------------------------------------------- deduplicate
        [V,~,idxMap] = uniquetol(C, tol, 'ByRows', true); %#ok<NASGU>

        %----------------------------------------------------- exact option
        ptsSym = []; Vol = [];
        if doExact
            ptsSym = sym(zeros(size(C)));
            for j = 1:numel(C)
                [p,q]     = rat(C(j), tol);
                ptsSym(j) = sym(p)/sym(q);
            end
        end

        %----------------------------------------------------- convex hull
        if size(V,1) >= 4 && rank(V) == 3
            [F, Vol] = convhull(V);            % these are all 3d, convhull() 
        else
            F = [];                      % degenerate
        end
        
        jsonFile = replace(files(k).name, '.txt', '.json');
        writeJSONsimp(fullfile(folder, jsonFile), F, V);   % omit V if you don’t need coords

        %------------------------------------------------ quality metrics
        q = qualityMetrics(V, F);

        %—— parse t-strength from the filename (des3-14-4.txt → t = 4)
        tok = regexp(files(k).name,'des\d+[-_]\d+[-_](\d+)','tokens','once');
        if ~isempty(tok)
            tFile = str2double(tok{1});
            q.maxMomentErr = strengthResidual(V, tFile, 1e-12);
        else
            q.maxMomentErr = NaN;
        end

        %----------------------------------------------------- data record
        data(k).file     = files(k).name;
        data(k).ptsFloat = C;
        data(k).ptsSym   = ptsSym;
        data(k).faces    = F;
        data(k).volume  = Vol;
        data(k).q        = q;

        %----------------------------------------------------- console log
        if isempty(F)
            fprintf('[%3d/%d] %-20s  pts=%3d  NO HULL                ', ...
                    k,nFiles,files(k).name,size(V,1));
        else
            fprintf('[%3d/%d] %-20s  F=%3d  EulerΔ=%+.0f  AR_max=%.2f  ', ...
                    k,nFiles,files(k).name,size(F,1),q.eulerDefect,q.AR_max);
        end
        if ~isnan(q.maxMomentErr)
            fprintf('Merr=%.1e\n', q.maxMomentErr);
        else
            fprintf('\n');
        end
    end
    fprintf('✓  Done.\n');
end
%======================================================================
function writeJSONsimp(name, faces, vertices)
    S = struct('facets', int32(faces-1));   % 0-based indices for Sage/GAP/M2
    if nargin > 2 && ~isempty(vertices)
        S.vertices = vertices;              % keep doubles exactly
    end
    fid = fopen(name, 'w');
    fprintf(fid, '%s', jsonencode(S, 'PrettyPrint',true));
    fclose(fid);
end
%=======================================================================
function q = qualityMetrics(V, F)
% Few lightweight mesh/point-set indicators.

    q = struct('eulerDefect',NaN,'minEdge',NaN,'maxEdge',NaN, ...
               'minAngle',NaN,'AR_max',NaN);

    if isempty(F),  return,  end

    %—— Euler characteristic
    E = edgesFromFaces(F);
    q.eulerDefect = size(V,1) - size(E,1) + size(F,1) - 2;

    %—— edge lengths
    L          = vecnorm(V(E(:,1),:) - V(E(:,2),:), 2, 2);
    q.minEdge  = min(L);
    q.maxEdge  = max(L);

    %—— triangle aspect ratios (longest/shortest edge per face)
    AR = zeros(size(F,1),1);
    for i = 1:size(F,1)
        tri = V(F(i,:),:);
        d   = pdist(tri);                     % 3 distances
        AR(i) = max(d) / min(d);
    end
    q.AR_max = max(AR);

    %—— minimum internal angle (quick)
    q.minAngle = minAngleInMesh(V, F);
end
%----------------------------------------------------------------------
function E = edgesFromFaces(F)
    E = unique(sort([F(:,[1 2]); F(:,[2 3]); F(:,[3 1])],2),'rows');
end
%----------------------------------------------------------------------
function a = minAngleInMesh(V, F)
% Minimal triangle angle in degrees (uses pdist2; core MATLAB ≥ R2022a).
    a = 180;
    for i = 1:size(F,1)
        tri = V(F(i,:),:);
        D2  = pdist2(tri, tri).^2;            % squared lengths
        for j = 1:3
            u = mod(j,3)+1;  v = mod(j+1,3)+1;
            cosA = (D2(j,u) + D2(j,v) - D2(u,v)) / (2*sqrt(D2(j,u)*D2(j,v)));
            a    = min(a, acosd(max(-1,min(1,cosA))));
        end
    end
end
%----------------------------------------------------------------------
function maxErr = strengthResidual(P, t, tol)
% Verify spherical t-design condition: moments up to degree t.
% Returns worst absolute error.  Pass if maxErr < tol.

    if nargin < 3, tol = 1e-15; end
    P = P ./ vecnorm(P,2,2);        % robust normalisation
    maxErr = 0;

    g  = arrayfun(@(n) gamma((n+1)/2), 0:t);
    g0 = 1/(4*pi);

    for i = 0:t
        Xi = P(:,1).^i;
        for j = 0:(t-i)
            Xj = P(:,2).^j;
            for k = 0:(t-i-j)
                mPts = mean(Xi .* Xj .* P(:,3).^k);        % design moment
                % old (wrong) --------------------
                % if mod(i|j|k,2)    % any non‑zero exponent triggers this!
                
                % new (correct) ------------------
                if mod(i,2) || mod(j,2) || mod(k,2)
                    mSphere = 0;

                else
                    mSphere = g0*2* g(i+1)*g(j+1)*g(k+1) / gamma((i+j+k+3)/2);
                end
                maxErr = max(maxErr, abs(mPts - mSphere));
            end
        end
    end

    if maxErr < tol
        % Silent pass; console output handled by caller
    end
end
