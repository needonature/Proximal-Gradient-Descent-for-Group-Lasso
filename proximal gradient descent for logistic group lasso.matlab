%proximal gradient descent for logistic group lasso
load('moviesGroups.mat');
load('moviesTest.mat');
load('moviesTrain.mat');
f_diff=[]
ff=[]
Y=trainLabels;
X=trainRatings;
[a,b]=size(X);
X=[ones(a,1),X];
t=10^-4;
lamda=5;
f_star=336.207;
beta=zeros(1,b+1);
f_diff=[]

for i = 1:1000 %%%
    i
    g=sum(repmat(exp(X*beta')./(1+exp(X*beta'))-Y,1,834).*X,1);
    beta1=beta-t*g;
    h_beta=0;
    for j=2:20
        p(j)=length(find(groupLabelsPerRating==(j-1)));
        beta1_group{j}=beta1(1+find(groupLabelsPerRating==(j-1)));
        w{j}=sqrt(p(j));
        aa{j}=t*lamda*w{j};
        if norm(beta1_group{j})<=aa{j}
            beta2_group{j}=0;
        else
            beta2_group{j}=beta1_group{j}/(aa{j}/(norm(beta1_group{j})-aa{j})+1);
            %compute h_beta
            h_beta=h_beta+lamda*w{j}*norm(beta2_group{j});
        end
        beta2(find(groupLabelsPerRating==(j-1))+1)=beta2_group{j};
    end
    beta2(1)=beta1(1);
    beta=beta2;
    %f(k)
    g_beta=-sum(Y.*(X*beta'))+sum(log(1+exp(X*beta')));
    f_k=g_beta+h_beta;
    ff=[ff,f_k];
    f_diff=[f_diff,log(f_k-f_star)];
end
k=1:1000
plot(k,f_diff(1:1000))
