%proximal gradient descent for least square group lasso
X=csvread('X.csv',1,0)
Y=csvread('Y.csv',1,0)
[a,b]=size(X)
beta=zeros(1,1+b)
p=[3,3,2,1,2,1,1,3]
t=0.002
lamda=4;
f_star=84.6952
X=[ones(a,1),X]

f_diff=[]
for i = 1:1000 %%%
    i
    g=-2*X'*(Y-X*beta');
    beta1=beta-t*g';
    h_beta=0;
    beta1_group{1}=beta1(2:4);
    beta1_group{2}=beta1(5:7);
    beta1_group{3}=beta1(8:9);
    beta1_group{4}=beta1(10);
    beta1_group{5}=beta1(11:12);
    beta1_group{6}=beta1(13);
    beta1_group{7}=beta1(14);
    beta1_group{8}=beta1(15:17);
    for j=1:8
        w{j}=sqrt(p(j));
        aa{j}=t*lamda*w{j};
        if norm(beta1_group{j})<=aa{j}
            beta2_group{j}=0;
        else
            beta2_group{j}=beta1_group{j}/(aa{j}/(norm(beta1_group{j})-aa{j})+1);
            %compute h_beta
            h_beta=h_beta+lamda*w{j}*norm(beta2_group{j});
        end
    end
    beta2(1)=beta1(1);
    beta2(2:4)=beta2_group{1};
    beta2(5:7)=beta2_group{2};
    beta2(8:9)=beta2_group{3};
    beta2(10)=beta2_group{4};
    beta2(11:12)=beta2_group{5};
    beta2(13)=beta2_group{6};
    beta2(14)=beta2_group{7};
    beta2(15:17)=beta2_group{8};
    beta=beta2;
    %f(k)
    g_beta=norm(Y-X*beta')^2;
    f_k=g_beta+h_beta;
    f_diff=[f_diff,log(f_k-f_star)];
end
k=1:1000
plot(k,f_diff(1:1000))
