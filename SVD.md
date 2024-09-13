~~~matlab
% 1) No preprocessing
A = data; 

% 2) Make mean = 0 in each column
colmeans = mean(data,1);
A = data - repmat(colmeans,nrpat,1);  % mean=0 in each column

% 3) Make standard deviation = 1 for each column
% colstds = std(A,1);
% A = A ./ repmat(colstds,nrpat,1);     % std = 1 in each column

[U,S,V]= svd(A,'econ'); % A m*n   A = USV'

%左奇异向量（U）： U是一个m×m的正交矩阵，其列向量被称为左奇异向量。
% 左奇异向量提供了关于矩阵A的行空间的信息。具体来说，U的列向量是矩阵A的特征向量，它们描述了A的行空间的正交基。
% 这些特征向量使得U成为一个正交矩阵，即其列向量两两正交且单位长度。

%奇异值（S）： S是一个m×m的对角矩阵，其对角线上的元素被称为奇异值。奇异值提供了关于矩阵A的奇异性和重要性的信息。
% 奇异值是矩阵A的奇异值分解的核心，它们描述了矩阵A的奇异性和重要性，是矩阵A的奇异向量的长度。

%右奇异向量（V）： V是一个n×m的正交矩阵，其列向量被称为右奇异向量。右奇异向量提供了关于矩阵A的列空间的信息。
% 具体来说，V的列向量是矩阵A的特征向量，它们描述了A的列空间的正交基。这些特征向量使得V成为一个正交矩阵，即其列向量两两正交且单位长度。
plot(A(indT,:)*V(:,1), A(indT,:)*V(:,2),'bx')
hold on
plot(A(indB,:)*V(:,1), A(indB,:)*V(:,2),'ro')
~~~

