function [Hstar,iter, obj] = pmksclustering(KH,k,lambda,beta,gamma,Y,numclass)

num = size(KH, 1); %the number of samples
numker = size(KH, 3); %m represents the number of kernels
maxIter = 150; %the number of iterations
H = zeros(num,k,numker);
G = zeros(num,num,numker);
Z = zeros(num,num);
opt.disp = 0;

flag = 1;
iter = 0;

while flag
    iter = iter +1;
    
    %% the first step-- optimize H_i
    for p=1:numker  
        Gp = G(:,:,p);
        A = KH(:,:,p)+lambda*(Gp+Gp'-eye(num)-Gp'*Gp);
        [AP,~] = eigs(A, k, 'la', opt);
        H(:,:,p) = AP;
    end
  
    %%
    %the second step-- optimize S_i
    for p=1:numker
        K =  H(:,:,p)*H(:,:,p)';
        tmp = (beta*Z + lambda*K)/(lambda*K + beta*eye(num));
        for ii=1:num
            idx = 1:num;
            idx(ii) = [];
            G(ii,idx,p) = EProjSimplex_new(tmp(ii,idx));
        end
    end
    %     %%
    
    %%
    %the third step-- optimize S
    
    Z = (beta/(numker*beta+gamma))*sum(G,3);
  
    
    %%
    term1 =0;
    term2 =0;
    term3 = 0;
    for j =1:numker
        term1 = term1+ trace(KH(:,:,j)-KH(:,:,j)*H(:,:,j)*H(:,:,j)');
        term2 = term2+ lambda*norm((H(:,:,j)-G(:,:,j)*H(:,:,j)),'fro')^2;
        term3 = term3 + beta*norm(G(:,:,j)-Z,'fro')^2;
    end
   
    term4 = gamma*norm(Z,'fro')^2;
    obj(iter) = term1+term2+term3+term4;
    
    
   
        if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter)
        Z_= (Z+Z')/2;
        Z_ =Z_-diag(Z_);
        [Hstar,~] = eigs(Z_, k, 'la', opt);
        %         disp(Hstar);
        %         mappedX = tsne(Hstar,'Perplexity',80);  %Y
        %         figure;
        %         gscatter(mappedX(:,1), mappedX(:,2),Y); axis off;        
        %     if iter==maxIter
        flag =0;
    end
end
