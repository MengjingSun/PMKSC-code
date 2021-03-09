clear
clc
warning off;

DataName{1} = 'YALE';


lambdaset = 2.^[-5:2:5];
betaset = 10.^[-5:2:5];
gammaset = 10.^[-5:2:5];


DataHyperParam{10} = [
    6,5,5;
    ];

count = 1;
for i=1:6
    for j=1:6
        for k=1:6
            allsort(count,:) = [i,j,k];
            count = count + 1;
        end
    end
end

DataHyperParam{1} = allsort;

;

path = './';
addpath(genpath(path));
k=0;
for i=[1]
    k=k+1;
    dataName = DataName{i} %%% flower17; flower102; proteinFold,caltech101_mit,UCI_DIGIT,ccv
    dataHyperParam = DataHyperParam{i};
    load([path,'datasets/',dataName,'_Kmatrix'],'KH','Y');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    numclass = length(unique(Y));
    numker = size(KH,3);
    num = size(KH,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    KH = kcenter(KH);
    KH = knorm(KH);
    K0 = zeros(num,num); 
    qnorm = 2;
    opt.disp = 0;
    tic
    
    for param=1:size(dataHyperParam,1)
        it = dataHyperParam(param,1);
        ij = dataHyperParam(param,2);
        iq = dataHyperParam(param,3);
        disp(['p1=',num2str(lambdaset(it)), '  p2=',num2str(betaset(ij)), '  p3=',num2str(gammaset(iq))]);
        tic;
        
        [H_normalized9, iter, obj] = pmksclustering(KH,numclass,lambdaset(it),betaset(ij),gammaset(iq),Y,numclass);
        res9 = myNMIACCwithmean(H_normalized9,Y,numclass);
        time(k)=toc;
        %
        accval9(it,ij,iq) = res9(1,1);      
        nmival9(it,ij,iq)= res9(2,1);      
        purval9(it,ij,iq) = res9(3,1);
    end
    res = [max(max(max(accval9))); max(max(max(nmival9)));max(max(max(purval9)))];
    save(['./',dataName,'-res.mat'],'res');
end


