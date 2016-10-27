%backtrack line search for logistic group lasso
load('moviesGroups.mat');
load('moviesTest.mat');
load('moviesTrain.mat');
f_diff=[];
Y=trainLabels;
X=trainRatings;
[a,b]=size(X);
X=[ones(a,1),X];
beta=zeros(1,b+1);
lamda=5;
f_star=336.207;
beta_back=0.1;
back_num=[];

for i = 1:400 %%%
    i

    %%proximal for backtrack
    t=1;
    g=sum(repmat(exp(X*beta')./(1+exp(X*beta'))-Y,1,834).*X,1);
    beta1=beta-t*g;
    for j=2:20
        p(j)=length(find(groupLabelsPerRating==(j-1)));
        beta1_group{j}=beta1(1+find(groupLabelsPerRating==(j-1)));
        w{j}=sqrt(p(j));
        aa{j}=t*lamda*w{j};
        if norm(beta1_group{j})<=aa{j}
            beta2_group{j}=0;
        else
            beta2_group{j}=beta1_group{j}/(aa{j}/(norm(beta1_group{j})-aa{j})+1);
        end
        beta2(find(groupLabelsPerRating==(j-1))+1)=beta2_group{j};
    end
    beta2(1)=beta1(1);

    %%backtrack line search
    G=(beta-beta2)/t;
    g_beta2=-sum(Y.*(X*beta2'))+sum(log(1+exp(X*beta2'))); %%%% really large g(beta2) ???
    g_beta=-sum(Y.*(X*beta'))+sum(log(1+exp(X*beta')));
    back_num0=0;
    g_beta2-(g_beta-t*g*G'+t/2*(norm(G)^2))
    while (g_beta2>(g_beta-t*g*G'+t/2*(norm(G)^2)))
        back_num0=back_num0+1;
        t=beta_back*t;
    end
    back_num=[back_num,back_num0];

    %%proximal
    h_beta=0;
    g=sum(repmat(exp(X*beta')./(1+exp(X*beta'))-Y,1,834).*X,1);
    beta1=beta-t*g;
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
    f_diff=[f_diff,log(f_k-f_star)];
end

update_point=[]
update_point0=0;
for ii=1:400
    update_point0=update_point0+back_num(ii);
    update_point(ii)=update_point0;
end
plot(update_point,f_diff)


