% Two-Dimensional PCA - Project 8
% Chandni Shrivastava
% Sachin Sundar Pungampalayam Shanmugasundaram

% Read images into data array
data_dir=dir('C:\ASU\FSL\Project\code\yalefaces');
data=[];
no_folders=length(data_dir)-2;
folders=data_dir([data_dir.isdir]);
no_files=length(dir([folders(3).folder '\' folders(3).name]))-2;
for i=1:no_folders
    for j=1:no_files
        bmp_file=imread(sprintf('C:/ASU/FSL/Project/code/yalefaces/%02d/s%d.bmp', i, j));
        data(:,:,j,i)=imresize(bmp_file, 0.5);
    end
end
[height,width,files,folders]=size(data);
data=reshape(data,[height,width,files*folders]);
label=kron([1:folders]',ones(files,1));
data_size=[height,width,files,folders];
rng(0);
[~,~,n]=size(data);
random=randperm(n);

% Create train and test set for classification
test_set=random(1:floor(n/10));
test_length=length(test_set);
test=data(:,:,test_set);
test_label=label(test_set);
train_set=setdiff([1:n],test_set);
train_length=length(train_set);
train=data(:,:,train_set);
train_label=label(train_set);
train_mean=mean(train,3);
train=train-repmat(train_mean,[1,1,train_length]);
test=test-repmat(train_mean,[1,1,test_length]);
train_size=size(train,1);
train_length=length(train_label);
test_length=length(test_label);
% Do PCA on the training set
[~,width,p]=size(train);
cov=zeros(width);
for i=1:p
    cov=cov+data(:,:,i)'*data(:,:,i);
end
[U,s]=eig(cov);
[~,index]=sort(abs(diag(s)),'descend');
U=U(:,index);
latent_features=30;
% Eigen vectors of image matrices
eigen_vectors=U(:,1:latent_features);
eigen_count=size(eigen_vectors,2);
train_eigen_values=zeros(train_size,eigen_count,train_length);
% Finding the Principal Components of image matrices
for i=1:train_length
    train_eigen_values(:,:,i)=train(:,:,i)*eigen_vectors;
end
test_eigen_values=zeros(train_size,eigen_count,test_length);
for i=1:test_length
    test_eigen_values(:,:,i)=test(:,:,i)*eigen_vectors;
end
for i=1:eigen_count
    train=train_eigen_values(:,1:i,:);
    test=test_eigen_values(:,1:i,:);
    train=reshape(train,numel(train)/train_length,train_length);
    test=reshape(test,numel(test)/test_length,test_length);   
    distance=pdist2(train',test');
    [~,n]=min(distance);
    acc(i)=mean(test_label==train_label(n));
end
figure;
kSet=[1:30]';
plot(kSet,acc);
pos=get(gcf,'Position');

scale=1.2;
title('Classification Accuracy');
xlabel('Number of extracted basis vectors');
ylabel('Classification accuracy');
% Add noise to images
kSet=[1:30];
[height,width,nSub]=size(data);
ix_noise=randperm(nSub);
p=nSub*20/100;
m=randi([20,height],p,1);
n=randi([20,width],p,1);
for i=1:p
    noise=255*randi([0,1],[m(i),n(i)]);
    posh=randi([1,height-m(i)+1]);
    posw=randi([1,width-n(i)+1]);
    data(posh:posh+m(i)-1, posw:posw+n(i)-1, ix_noise(i))=noise;
end
noise_data=data;

nSub=size(noise_data,3);
noise_mean=mean(noise_data,3);
x_centered=noise_data-repmat(noise_mean,[1,1,nSub]);
[~,width,p]=size(x_centered);
cov=zeros(width);
for i=1:p
    cov=cov+data(:,:,i)'*data(:,:,i);
end
[U,s]=eig(cov);
[~,index]=sort(abs(diag(s)),'descend');
U=U(:,index);
latent_features=30;
W=U(:,1:latent_features);
% Reconstruct the image using Principal Components and eigen vectors
nK=length(kSet);
xv_reco=zeros(size(noise_data));
err=zeros(nK,1);
tic;
for iK=1:nK
    w=W(:,1:kSet(iK));
    for iSub=1:nSub
        x_reco(:,:,iSub)=x_centered(:,:,iSub)*w*w'+noise_mean;
    end
    temp=noise_data-x_reco(:,:,:);
    
    sum=0;
    for iSub=nSub/5+1:nSub
        sum=sum+norm(temp(:,:,ix_noise(iSub)),'fro');
    end
    err(iK)=sum/(nSub/5*4);
end
figure;
plot(kSet,err,'-o');
title('Reconstruction Accuracy');
xlabel('Number of extracted features');
ylabel('Average reconstruction error');
