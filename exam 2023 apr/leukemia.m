% Leukemia data for 130 patients
% A(indT,:) matrix with 13 patients * 22282 measurements, type T cancer
% A(indB,:) matrix with 117 patients* 22282 measurements, type B cancer

clear;  close all; rng(2)
load cancer

indT = 1:13;    % Leukemia type T
indB = 14:130;  % Leukemia type B
nrpat = 130;
nrmeas = 22282;

%% Choose best way to preprocess this data, the choice is not obvious

% 1) No preprocessing
A = data; 

% 2) Make mean = 0 in each column
colmeans = mean(data,1);
A = data - repmat(colmeans,nrpat,1);  % mean=0 in each column

% 3) Make standard deviation = 1 for each column
% colstds = std(A,1);
% A = A ./ repmat(colstds,nrpat,1);     % std = 1 in each column

% % 4) Combination of 2 and 3
% colstds = std(A,1);
% A = A ./ repmat(colstds,nrpat,1);
% colmeans = mean(data,1);
% A = A - repmat(colmeans,nrpat,1);

%% SVD decomposition of A

[U,S,V]= svd(A,'econ'); % A m*n   A = USV'

%左奇异向量（U）： U是一个m×m的正交矩阵，其列向量被称为左奇异向量。
% 左奇异向量提供了关于矩阵A的行空间的信息。具体来说，U的列向量是矩阵A的特征向量，它们描述了A的行空间的正交基。
% 这些特征向量使得U成为一个正交矩阵，即其列向量两两正交且单位长度。

%奇异值（S）： S是一个m×m的对角矩阵，其对角线上的元素被称为奇异值。奇异值提供了关于矩阵A的奇异性和重要性的信息。
% 奇异值是矩阵A的奇异值分解的核心，它们描述了矩阵A的奇异性和重要性，是矩阵A的奇异向量的长度。

%右奇异向量（V）： V是一个n×m的正交矩阵，其列向量被称为右奇异向量。右奇异向量提供了关于矩阵A的列空间的信息。
% 具体来说，V的列向量是矩阵A的特征向量，它们描述了A的列空间的正交基。这些特征向量使得V成为一个正交矩阵，即其列向量两两正交且单位长度。

%% Plot patient data in 2D using SVD

figure(1)
% Tx = randn(nrmeas,1);   % Replace these transformations. Use the SVD somehow.
% Ty = randn(nrmeas,1);
% plot(A(indT,:)*Tx, A(indT,:)*Ty,'bx','markersize',8); hold on
% plot(A(indB,:)*Tx, A(indB,:)*Ty,'ro','markersize',8)

colmeans = mean(data,1);
% A = data - ones(nrpat,1) * colmeans;
plot(A(indT,:)*V(:,1), A(indT,:)*V(:,2),'bx')
hold on
plot(A(indB,:)*V(:,1), A(indB,:)*V(:,2),'ro')

%% Classify patient1 and patient2  (vectors with 1*22282 numbers)
%  from the information seen in the figure

Tx=(V(:,1));
Ty=(V(:,2));
colmeans = mean(data,1);
pat1 = patient1 - colmeans;
pat2 = patient2 - colmeans;

plot(pat1*V(:,1) , pat1*V(:,2),'k*','markersize',16)
plot(pat2*V(:,1) , pat2*V(:,2),'k+','markersize',16)
legend('type T','type B','patient1','patient2')
% %%
% pat1 = patient1 - colmeans;
% pat2 = patient2 - colmeans;
% plot(pat1*V(:,1) , pat1*V(:,2),'k*','markersize',16)
% plot(pat2*V(:,1) , pat2*V(:,2),'k+','markersize',16)
% legend('type T','type B','patient1','patient2')
