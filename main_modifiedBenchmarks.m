% This is the modified code for WCCI 2018 MTOE competition
% Dongrui Wu (drwu@hust.edu.cn), 3/20/2018
% For maximization problems, multiply objective function by -1.
%
% Settings of simulated binary crossover (SBX) in this code is Pc = 1,
% and probability of variable sawpping = 0.
%

clc; clearvars; close all; rng('default'); %warning off all;
popSize=200; % Population size
nGen=500; % Number of generations
selProcess = 'elitist'; % Choose either 'elitist' or 'roulette wheel'
pIL = 0; % Probability of individual learning (BFGA quasi-Newton Algorithm) --> Indiviudal Learning is an IMPORTANT component of the MFEA.
rmp=0.3; % Random mating probability
nRepeat = 20; % Number of repeats; should be 30 in final submission; must >1 to avoid no-display problem
pTransfer=0.4; % Portion of chromosomes to transfer from one task to another
eMin=0.01; % Threshold of accumulated survival rate of divergents

dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(data) fprintf('%d-%s-%d ', data{1},data{2},data{3})); % print progress of parfor
dqClient=parallel.pool.DataQueue; afterEach(dqClient,@showResults);

parfor idxTask = 1:9 % parallel execution
    tasks = benchmarkModified(idxTask); initPop=cell(2,nRepeat);
    SOEA1=SOEA(tasks(1),popSize/length(tasks),nGen,selProcess,pIL,nRepeat,idxTask,dqWorker);
    SOEA2=SOEA(tasks(2),popSize/length(tasks),nGen,selProcess,pIL,nRepeat,idxTask,dqWorker);
    for r=1:nRepeat
        initPop{1,r}=SOEA1.initPop{r}; initPop{2,r}=SOEA2.initPop{r}; % initial population
    end
    data2(idxTask)=MFEA(tasks,popSize,nGen,selProcess,rmp,pIL,nRepeat,idxTask,dqWorker,initPop); % The provided MFEA benchmark algorithm
    data1(idxTask)=data2(idxTask);
    data1(idxTask).wallClockTime=SOEA1.wallClockTime+SOEA2.wallClockTime;
    data1(idxTask).bestFitness=cat(1,SOEA1.bestFitness,SOEA2.bestFitness);
    data1(idxTask).bestIndData=[SOEA1.bestIndData; SOEA2.bestIndData];
    data1(idxTask).totalEvals=SOEA1.totalEvals+SOEA2.totalEvals;
    data3(idxTask)=MFEARR(tasks,popSize,nGen,selProcess,rmp,pIL,nRepeat,idxTask,dqWorker,eMin,initPop);
    data4(idxTask)=LDAMFEA(tasks,popSize,nGen,selProcess,rmp,pIL,nRepeat,idxTask,dqWorker,initPop);
    data5(idxTask)=MFEALBS(tasks,popSize,nGen,selProcess,rmp,pIL,nRepeat,idxTask,dqWorker,initPop);
    data6(idxTask)=EBSGA(tasks,popSize/length(tasks),nGen,selProcess,rmp,pIL,nRepeat,idxTask,dqWorker,initPop);
    data7(idxTask)=GMFEA(tasks,popSize,nGen,selProcess,rmp,pIL,nRepeat,idxTask,dqWorker,initPop);
    data8(idxTask)=EMTEA(tasks,popSize/length(tasks),nGen,selProcess,pIL,nRepeat,idxTask,dqWorker,initPop);
    data9(idxTask)=MTEAbest(tasks,popSize/length(tasks),nGen,selProcess,pIL,nRepeat,pTransfer,idxTask,dqWorker,initPop); % Our algorithm, v2
    
    data=struct('wallClockTime1',data1(idxTask).wallClockTime,'bestFitness1',data1(idxTask).bestFitness,...
        'bestIndData1',data1(idxTask).bestIndData,'totalEvals1',data1(idxTask).totalEvals,...
        'wallClockTime2',data2(idxTask).wallClockTime,'bestFitness2',data2(idxTask).bestFitness,...
        'bestIndData2',data2(idxTask).bestIndData,'totalEvals2',data2(idxTask).totalEvals,...
        'wallClockTime3',data3(idxTask).wallClockTime,'bestFitness3',data3(idxTask).bestFitness,...
        'bestIndData3',data3(idxTask).bestIndData,'totalEvals3',data3(idxTask).totalEvals,...
        'wallClockTime4',data4(idxTask).wallClockTime,'bestFitness4',data4(idxTask).bestFitness,...
        'bestIndData4',data4(idxTask).bestIndData,'totalEvals4',data4(idxTask).totalEvals,...
        'wallClockTime5',data5(idxTask).wallClockTime,'bestFitness5',data5(idxTask).bestFitness,...
        'bestIndData5',data5(idxTask).bestIndData,'totalEvals5',data5(idxTask).totalEvals,...
        'wallClockTime6',data6(idxTask).wallClockTime,'bestFitness6',data6(idxTask).bestFitness,...
        'bestIndData6',data6(idxTask).bestIndData,'totalEvals6',data6(idxTask).totalEvals,...
        'wallClockTime7',data7(idxTask).wallClockTime,'bestFitness7',data7(idxTask).bestFitness,...
        'bestIndData7',data7(idxTask).bestIndData,'totalEvals7',data7(idxTask).totalEvals,...
        'wallClockTime8',data8(idxTask).wallClockTime,'bestFitness8',data8(idxTask).bestFitness,...
        'bestIndData8',data8(idxTask).bestIndData,'totalEvals8',data8(idxTask).totalEvals,...
        'wallClockTime9',data9(idxTask).wallClockTime,'bestFitness9',data9(idxTask).bestFitness,...
        'bestIndData9',data9(idxTask).bestIndData,'totalEvals9',data9(idxTask).totalEvals,...
        'idxTask',idxTask);
    send(dqClient,data);
%     parSave(['results_' num2str(idxTask) '.mat'],data1(idxTask),data2(idxTask),data3(idxTask),...
%         data4(idxTask),data5(idxTask),data6(idxTask),data7(idxTask),data8(idxTask),data9(idxTask),nGen,nRepeat);
end
save('resultsModified9.mat','data1', 'data2', 'data3','data4', 'data5', 'data6','data7','data8','data9','nGen','nRepeat');
plotAllResults;


% function parSave(fname,data1,data2, data3,data4,data5, data6,data7,data8,data9,nGen,nRepeat)
% save(fname,'data1', 'data2', 'data3','data4', 'data5', 'data6','data7','data8','data9','nGen','nRepeat');
% end

% Display results in parfor
function showResults(data)
nRepeat=size(data.bestFitness1,1)/2;
fitness=[mean(data.bestFitness1(1:nRepeat,:))' mean(data.bestFitness1(nRepeat+1:end,:))' ... 
    mean(data.bestFitness2(1:2:end,:))' mean(data.bestFitness2(2:2:end,:))' ...
    mean(data.bestFitness3(1:2:end,:))' mean(data.bestFitness3(2:2:end,:))' ...
    mean(data.bestFitness4(1:2:end,:))' mean(data.bestFitness4(2:2:end,:))' ...
    mean(data.bestFitness5(1:2:end,:))' mean(data.bestFitness5(2:2:end,:))' ...
    squeeze(mean(data.bestFitness6,1)) ...
    mean(data.bestFitness7(1:2:end,:))' mean(data.bestFitness7(2:2:end,:))' ...
    squeeze(mean(data.bestFitness8,1)) squeeze(mean(data.bestFitness9,1))];
[fitness(end,1:2)   data.wallClockTime1; ...
    fitness(end,3:4)   data.wallClockTime2; ...
    fitness(end,5:6)   data.wallClockTime3; ...
    fitness(end,7:8)   data.wallClockTime4; ...
    fitness(end,9:10)   data.wallClockTime5; ...
    fitness(end,11:12)   data.wallClockTime6; ...
    fitness(end,13:14) data.wallClockTime7;...
    fitness(end,15:16) data.wallClockTime8;...
    fitness(end,17:18) data.wallClockTime9];
figure;
subplot(121);
semilogy(data.totalEvals1(1,:),fitness(:,1),'k-','linewidth',2); hold on;
semilogy(data.totalEvals2(1,:),fitness(:,3),'k--','linewidth',2);
semilogy(data.totalEvals3(1,:),fitness(:,5),'r--','linewidth',2);
semilogy(data.totalEvals4(1,:),fitness(:,7),'g--','linewidth',2);
semilogy(data.totalEvals5(1,:),fitness(:,9),'b--','linewidth',2);
semilogy(data.totalEvals6(1,:),fitness(:,11),'r-','linewidth',2);
semilogy(data.totalEvals7(1,:),fitness(:,13),'g-','linewidth',2);
semilogy(data.totalEvals8(1,:),fitness(:,15),'b-','linewidth',2);
semilogy(data.totalEvals9(1,:),fitness(:,17),'k:','linewidth',2);axis tight;
legend('SOEA','MFEA','MFEARR','LDAMFEA','MFEALBS','EBSGA','GMFEA','EMTEA','MTEA','location','northeast');
title(['Benchmark ' num2str(data.idxTask) ', Task 1']);
xlabel('# func. eval.'); ylabel('Objective');
subplot(122);
semilogy(data.totalEvals1(1,:),fitness(:,2),'k-','linewidth',2); hold on;
semilogy(data.totalEvals2(1,:),fitness(:,4),'k--','linewidth',2);
semilogy(data.totalEvals3(1,:),fitness(:,6),'r--','linewidth',2);
semilogy(data.totalEvals4(1,:),fitness(:,8),'g--','linewidth',2);
semilogy(data.totalEvals5(1,:),fitness(:,10),'b--','linewidth',2);
semilogy(data.totalEvals6(1,:),fitness(:,12),'r-','linewidth',2);
semilogy(data.totalEvals7(1,:),fitness(:,14),'g-','linewidth',2);
semilogy(data.totalEvals8(1,:),fitness(:,16),'b-','linewidth',2);
semilogy(data.totalEvals9(1,:),fitness(:,18),'k:','linewidth',2);axis tight;
legend('SOEA','MFEA','MFEARR','LDAMFEA','MFEALBS','EBS','GMFEA','EMTEA','MTEA','location','northeast');
title(['Benchmark ' num2str(data.idxTask) ', Task 2']);
xlabel('# func. eval.'); ylabel('Objective'); drawnow;
end

% Plot all results
function plotAllResults()
load resultsModified9; close all;
for idxTask = 1:9
    fitness=[mean(data1(idxTask).bestFitness(1:nRepeat,:))' mean(data1(idxTask).bestFitness(nRepeat+1:end,:))' ...
        mean(data2(idxTask).bestFitness(1:2:end,:))' mean(data2(idxTask).bestFitness(2:2:end,:))' ...
        mean(data3(idxTask).bestFitness(1:2:end,:))' mean(data3(idxTask).bestFitness(2:2:end,:))' ...
        mean(data4(idxTask).bestFitness(1:2:end,:))' mean(data4(idxTask).bestFitness(2:2:end,:))' ...
        mean(data5(idxTask).bestFitness(1:2:end,:))' mean(data5(idxTask).bestFitness(2:2:end,:))' ...
        squeeze(mean(data6(idxTask).bestFitness,1)) ...
        mean(data7(idxTask).bestFitness(1:2:end,:))'  mean(data7(idxTask).bestFitness(2:2:end,:))' ...
        squeeze(mean(data8(idxTask).bestFitness,1))   squeeze(mean(data9(idxTask).bestFitness,1))];
    [fitness(end,1:2)   data1(idxTask).wallClockTime; ...
        fitness(end,3:4)   data2(idxTask).wallClockTime; ...
        fitness(end,5:6)   data3(idxTask).wallClockTime; ...
        fitness(end,7:8)   data4(idxTask).wallClockTime; ...
        fitness(end,9:10)   data5(idxTask).wallClockTime; ...
        fitness(end,11:12)   data6(idxTask).wallClockTime; ...
        fitness(end,13:14) data7(idxTask).wallClockTime; ...
        fitness(end,15:16) data8(idxTask).wallClockTime; ...
        fitness(end,17:18) data9(idxTask).wallClockTime]
    figure;
    subplot(121);
    semilogy(data1(idxTask).totalEvals(1,:),fitness(:,1),'k-','linewidth',2); hold on;
    semilogy(data2(idxTask).totalEvals(1,:),fitness(:,3),'k--','linewidth',2);
    semilogy(data3(idxTask).totalEvals(1,:),fitness(:,5),'r--','linewidth',2);
    semilogy(data4(idxTask).totalEvals(1,:),fitness(:,7),'g--','linewidth',2);
    semilogy(data5(idxTask).totalEvals(1,:),fitness(:,9),'b--','linewidth',2);
    semilogy(data6(idxTask).totalEvals(1,:),fitness(:,11),'r-','linewidth',2);
    semilogy(data7(idxTask).totalEvals(1,:),fitness(:,13),'g-','linewidth',2);
    semilogy(data8(idxTask).totalEvals(1,:),fitness(:,15),'b-','linewidth',2);
    semilogy(data9(idxTask).totalEvals(1,:),fitness(:,17),'k:','linewidth',2);axis tight;
    legend('SOEA','MFEA','MFEARR','LDAMFEA','MFEALBS','EBSGA','GMFEA','EMTEA','MTEA','location','northeast');
    title(['Benchmark ' num2str(idxTask) ', Task 1']);
    xlabel('# func. eval.'); ylabel('Objective');
    subplot(122);
    semilogy(data1(idxTask).totalEvals(1,:),fitness(:,2),'k-','linewidth',2); hold on;
    semilogy(data2(idxTask).totalEvals(1,:),fitness(:,4),'k--','linewidth',2);
    semilogy(data3(idxTask).totalEvals(1,:),fitness(:,6),'r--','linewidth',2);
    semilogy(data4(idxTask).totalEvals(1,:),fitness(:,8),'g--','linewidth',2);
    semilogy(data5(idxTask).totalEvals(1,:),fitness(:,10),'b--','linewidth',2);
    semilogy(data6(idxTask).totalEvals(1,:),fitness(:,12),'r-','linewidth',2);
    semilogy(data7(idxTask).totalEvals(1,:),fitness(:,14),'g-','linewidth',2);
    semilogy(data8(idxTask).totalEvals(1,:),fitness(:,16),'b-','linewidth',2);
    semilogy(data9(idxTask).totalEvals(1,:),fitness(:,18),'k:','linewidth',2);axis tight;
    legend('SOEA','MFEA','MFEARR','LDAMFEA','MFEALBS','EBSGA','GMFEA','EMTEA','MTEA','location','northeast');
    title(['Benchmark ' num2str(idxTask) ', Task 2']);
    xlabel('# func. eval.'); ylabel('Objective');
end

nAlgs=9;
nPoints=min(100,nGen);
s=nGen/nPoints;
score=nan(9,nAlgs,nPoints);
outX=s:s:nGen;
for idx = 1:9
    [tasks,g1,g2] = benchmarkModified(idx);
    %    The globally optimal objective function value known in advance
    BF1=tasks(1).fnc(g1);
    BF2=tasks(2).fnc(g2);
    BFEV1=[data1(idx).bestFitness(1:nRepeat,:);  data2(idx).bestFitness(1:2:end,:); ...
        data3(idx).bestFitness(1:2:end,:);  data4(idx).bestFitness(1:2:end,:); ...
        data5(idx).bestFitness(1:2:end,:);  squeeze(data6(idx).bestFitness(:,:,1)); ...
        data7(idx).bestFitness(1:2:end,:); squeeze(data8(idx).bestFitness(:,:,1)); ...
        squeeze(data9(idx).bestFitness(:,:,1))] ...
        - BF1*ones(nAlgs*nRepeat,nGen);
    
    BFEV2=[data1(idx).bestFitness(nRepeat+1:end,:); data2(idx).bestFitness(2:2:end,:); ...
        data3(idx).bestFitness(2:2:end,:); data4(idx).bestFitness(2:2:end,:); ...
        data5(idx).bestFitness(2:2:end,:); squeeze(data6(idx).bestFitness(:,:,2)); ...
        data7(idx).bestFitness(2:2:end,:); squeeze(data8(idx).bestFitness(:,:,2)); ...
        squeeze(data8(idx).bestFitness(:,:,2))] ...
        - BF2*ones(nAlgs*nRepeat,nGen);
    
    u1=mean(BFEV1(:,outX));
    u2=mean(BFEV2(:,outX));
    st1=std(BFEV1(:,outX));
    st2=std(BFEV2(:,outX));
    st1(st1==0)=1;
    st2(st2==0)=1;
    ST1=repmat(st1,nRepeat,1);
    ST2=repmat(st2,nRepeat,1);
    for i=1:nAlgs
        score(idx,i,:)=mean((BFEV1((i-1)*nRepeat+1 : i*nRepeat,outX)-u1)./ST1+(BFEV2((i-1)*nRepeat+1 : i*nRepeat,outX)-u2)./ST2);
    end
end

Score=squeeze(sum(score));
figure;
linestyle={'k-','k--','r--','g--','b--','r-','g-','b-','k:'};
hold on;
title('Overall performance');
for i=1:nAlgs
    plot(data1(1).totalEvals(1,outX),Score(i,:),linestyle{i},'linewidth',2);
end
axis tight;
legend('SOEA','MFEA','MFEARR','LDAMFEA','MFEALBS','EBSGA','GMFEA','EMTEA','MTEA','location','northeast');
xlabel('# func. eval.'); ylabel('score');
drawnow;
end