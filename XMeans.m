% 
% MATLAB Implementation of X-means
% 
% This code is a MATLAB implementation of the X-means originally implemented
% in Python by Y. GOTO. The original Python code can be found at:
% https://gist.github.com/yasaichi/254a060eff56a3b3b858
% 
% Reference:
% http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
% 
% Author: N. Masuyama
% Date: 19 August, 2024
% Contact: masuyama@omu.ac.jp
% 
% License: This MATLAB code is licensed under the MIT License.
% 
% Tested on MATLAB R2024a
% 
classdef XMeans
    properties
        k_init
        k_means_args
        clusters
        labels_
        cluster_centers_
        cluster_log_likelihoods_
        cluster_sizes_
    end
    
    methods
        function obj = XMeans(k_init, varargin)
            if nargin < 1
                obj.k_init = 2;
            else
                obj.k_init = k_init;
            end
            obj.k_means_args = varargin;
        end
        
        function obj = fit(obj, X)
            obj.clusters = {};
            
            [idx, centers] = kmeans(X, obj.k_init, obj.k_means_args{:});
            b_c = obj.buildClusters(X, idx, centers);
            obj = obj.recursively_split(b_c);
            
            obj.labels_ = zeros(size(X, 1), 1);
            for i = 1:length(obj.clusters)
                obj.labels_(obj.clusters{i}.index) = i;
            end
            
            % Extract cluster centers individually and concatenate them
            cluster_centers = cellfun(@(c) c.center, obj.clusters, 'UniformOutput', false);
            obj.cluster_centers_ = vertcat(cluster_centers{:});
            
            % Likelihood calculation
            obj.cluster_log_likelihoods_ = cellfun(@(c) XMeans.log_likelihood(c), obj.clusters);
            obj.cluster_sizes_ = cellfun(@(c) c.size, obj.clusters);
        end
        
        function obj = recursively_split(obj, clusters)
            for i = 1:length(clusters)
                cluster = clusters{i};
                if cluster.size <= 3
                    obj.clusters{end+1} = cluster;
                    continue;
                end
                
                [idx, centers] = kmeans(cluster.data, 2, obj.k_means_args{:});
                c1 = XMeans.buildCluster(cluster.data(idx == 1, :), centers(1, :), cluster.index(idx == 1));
                c2 = XMeans.buildCluster(cluster.data(idx == 2, :), centers(2, :), cluster.index(idx == 2));
                
                beta = norm(c1.center - c2.center) / sqrt(det(c1.cov) + det(c2.cov));
                alpha = 0.5 / normcdf(beta);
                bic = -2 * (cluster.size * log(alpha) + XMeans.log_likelihood(c1) + XMeans.log_likelihood(c2)) + 2 * cluster.df * log(cluster.size);
                
                if bic < XMeans.bic(cluster)
                    obj = obj.recursively_split({c1, c2});
                else
                    obj.clusters{end+1} = cluster;
                end
            end
        end
        
        function clusters = buildClusters(~, X, idx, centers)
            clusters = {};
            for label = 1:max(idx)
                clusters{end+1} = XMeans.buildCluster(X(idx == label, :), centers(label, :), find(idx == label));
            end
        end
    end
    
    methods(Static)
        function cluster = buildCluster(data, center, index)
            cluster.data = data;
            cluster.index = index;
            cluster.size = size(data, 1);
            cluster.df = size(data, 2) * (size(data, 2) + 3) / 2;
            cluster.center = center;
            cluster.cov = cov(data);
            % Add small values to diagonal elements for regularization
            cluster.cov = cluster.cov + 1e-6 * eye(size(cluster.cov));
        end
        
        function log_likelihood = log_likelihood(cluster)
            log_likelihood = sum(log(mvnpdf(cluster.data, cluster.center, cluster.cov)));
        end
        
        function bic = bic(cluster)
            bic = -2 * XMeans.log_likelihood(cluster) + cluster.df * log(cluster.size);
        end
    end
end
