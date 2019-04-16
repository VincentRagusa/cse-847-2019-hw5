%main code
DATA_POINTS = 500;
FEATURES = 2;
NUM_CLUSTERS = 3;
ALGORITHM_ITER = 100;

%generate fake data
data = normrnd(0,0.5,[DATA_POINTS, FEATURES]); %data is generated as 2D for graphing results
data = [data; normrnd(5,0.5,[DATA_POINTS, FEATURES])];
data = [data; normrnd(2.5,1,[DATA_POINTS, FEATURES])];

%find clusters
[ga, gl, cen] = kmeans(data, NUM_CLUSTERS, ALGORITHM_ITER);
disp(cen)
%graph clusters
figure
gscatter(data(:, 1), data(:, 2), gl)

%spectral relax the problem by transforming the data
data_relaxed = spectral_relaxation_pre_process(data, NUM_CLUSTERS);

%find clusters
[ga_r, gl_r, cen_r] = kmeans(data_relaxed, NUM_CLUSTERS, ALGORITHM_ITER);
% disp(cen_r)
%graph clusters
figure
gscatter(data(:, 1), data(:, 2), gl_r)

%graph transformed clusters
% figure
% gscatter(data_relaxed(:, 1), data_relaxed(:, 2), gl_r)


%def kmeans
function [group_assignment, group_labels, group_centroids] = kmeans(X,k,maxGens)
    %implimentation of k-means unsupervised clustering

    [n, f] = size(X); %collect the size of the data set
    C = rand(k,f); %init centroids randomly
    for gen=1:maxGens
        %assign each data point to the closest centroid
        G = zeros(n,k); %init all data points to be groupless
        for i=1:n %for each data point
            centroid_assignment = -1; %dummy value
            min_dist = 1.e1000; %Inf
            for j=1:k %for each centroid
                if norm(X(i)-C(j)) < min_dist %if this centroid is closer
                    min_dist = norm(X(i)-C(j)); %update distance
                    centroid_assignment = j; % and temp assignment
                end
            end
            G(i,centroid_assignment) = 1;
        end
        %update each centroid to be the center of the points assigned to it
        for j=1:k %for each centroid
            s1 = zeros(1,f); %init centroid to zero
            s2 = 0; %init normalization factor to zero
            for i=1:n %for each data point
                s1 = s1 + G(i,j)*X(i); % element-wise sum all points in group j
                s2 = s2 + G(i,j); %count the number of points in group j
            end
            C(j,:) = s1/s2; %set the centroid to group j's center of mass
        end
    end
    
    %find group labels for each group
    L = zeros(n,1);
    for i = 1:n
        for j = 1:k
            if G(i,j) == 1
                L(i) = j;
            end
        end
    end
    
    group_assignment = G;
    group_labels = L;
    group_centroids = C;
end


function [transformed_data] = spectral_relaxation_pre_process(X,k)
% transforms the input data matrix using SVD into a form that k-means 
% can more easily cluster

[U,S,V] = svd(X);
% disp(size(U)) % data x data
% disp(size(S)) % data x features
% disp(V) % features x features
% disp((U*S*V.')-X) %should be close to 0
[features, ~] = size(V);
if k >= features
   disp("WARNING! k is not less than the original number of features!");
end
clip = min(features, k);
truncated_V = V(1:clip, :);
% disp(truncated_V);
transformed_data = X * truncated_V;
end
