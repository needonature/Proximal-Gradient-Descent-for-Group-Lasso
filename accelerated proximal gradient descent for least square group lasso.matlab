%accelerated proximal gradient descent for least square group lasso
X=csvread('X.csv',1,0)
Y=csvread('Y.csv',1,0)
[a,b]=size(X)
beta=zeros(1,1+b)
p=[3,3,2,1,2,1,1,3]
t=0.002
lamda=4;
f_star=84.6952
X=[ones(a,1),X]
a_f_diff=[]
a_beta_=zeros(1,b+1)
a_beta=zeros(1,b+1)

for i = 1:1000 %%%
    i
    v=a_beta+(i-2)/(i+1)*(a_beta-a_beta_);
    g=-2*X'*(Y-X*v');
    a_beta1=v-t*g';
    h_a_beta=0;
    a_beta1_group{1}=a_beta1(2:4);
    a_beta1_group{2}=a_beta1(5:7);
    a_beta1_group{3}=a_beta1(8:9);
    a_beta1_group{4}=a_beta1(10);
    a_beta1_group{5}=a_beta1(11:12);
    a_beta1_group{6}=a_beta1(13);
    a_beta1_group{7}=a_beta1(14);
    a_beta1_group{8}=a_beta1(15:17);
    for j=1:8
        w{j}=sqrt(p(j));
        aa{j}=t*lamda*w{j};
        if norm(a_beta1_group{j})<=aa{j}
            a_beta2_group{j}=0;
        else
            a_beta2_group{j}=a_beta1_group{j}/(aa{j}/(norm(a_beta1_group{j})-aa{j})+1);
            %compute h_a_beta
            h_a_beta=h_a_beta+lamda*w{j}*norm(a_beta2_group{j});
        end
    end
    a_beta2(1)=a_beta1(1);
    a_beta2(2:4)=a_beta2_group{1};
    a_beta2(5:7)=a_beta2_group{2};
    a_beta2(8:9)=a_beta2_group{3};
    a_beta2(10)=a_beta2_group{4};
    a_beta2(11:12)=a_beta2_group{5};
    a_beta2(13)=a_beta2_group{6};
    a_beta2(14)=a_beta2_group{7};
    a_beta2(15:17)=a_beta2_group{8};
    a_beta_=a_beta;
    a_beta=a_beta2;
    %f(k)
    g_a_beta=norm(Y-X*a_beta')^2;
    f_k=g_a_beta+h_a_beta;
    a_f_diff=[a_f_diff,log(f_k-f_star)];
end
k=1:1000
plot(k,a_f_diff(1:1000))
