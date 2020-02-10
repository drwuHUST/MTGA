function dataMTEA = MTEA2(tasks,popSize,nGen,selPocess,pIL,nRepeat,pTransfer,idxTask,dq)
% MTEA, Dongrui WU (drwu@hust.edu.cn), 4/18/2018
% 1. use scale to change nTransfer according to the similarity between tasks;
% 2. do not copy good chromosomes from the previous task directly; replace bad chromosomes by good ones from the previous task in reproduction; cosider order and bias
% 3. remove duplicates
% 4. do not consider bias when gen>=600

tic;

dataDisp=cell(1,3);
dataDisp{1}=idxTask; dataDisp{2}='MTEA2';

mu = 1.9;     % Index of Simulated Binary Crossover (tunable)
mum = 5.5;    % Index of polynomial mutation

if mod(popSize,2)
    popSize = popSize + 1;
end

nTasks=length(tasks); dimTasks=zeros(1,nTasks); population=cell(1,nTasks);
for i=1:nTasks
    dimTasks(i) = tasks(i).dims;
end

callsPerIndividual=zeros(1,popSize);
bestFitness = zeros(nRepeat,nGen,nTasks);    % best fitness found
totalEvals=zeros(nRepeat,nGen);   % total number of task evaluations so far
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning
scale=ones(1,nTasks); % adjust nTransfer dynamically
nTransfer0=round(pTransfer*popSize);

for r = 1:nRepeat
    dataDisp{3}=r;
    dq.send(dataDisp);
    
    % Initialize the first generation of Task 1 randomly
    gen=1; idxTask=1;
    for i = 1 : popSize
        population{idxTask}(i) = Chromosome();
        population{idxTask}(i) = initialize(population{idxTask}(i),dimTasks(idxTask));
        [population{idxTask}(i),callsPerIndividual(i)] = evaluate_SOO(population{idxTask}(i),tasks(idxTask),pIL,options);
    end
    
    totalEvals(r,gen)=sum(callsPerIndividual);
    fCosts=[population{idxTask}.factorial_costs];
    [fCosts,idsCost]=sort(fCosts);
    population{idxTask}=population{idxTask}(idsCost); % sort the chromosomes according to their costs
    bestFitness(r,gen,idxTask)=fCosts(1);
    
    % Initialize the first generation of other tasks semi-randomly
    for idxTask=2:nTasks
        nTransfer=nTransfer0;
        for i = 1 : nTransfer % Transfer nTransfer chromosomes from the previous task
            population{idxTask}(i) = Chromosome();
            if dimTasks(idxTask)>dimTasks(idxTask-1) % the previous task has a smaller dimensionality
                population{idxTask}(i).rnvec(1:dimTasks(idxTask-1))=population{idxTask-1}(i).rnvec(1:dimTasks(idxTask-1));
                population{idxTask}(i).rnvec(1+dimTasks(idxTask-1):dimTasks(idxTask))=rand(1,dimTasks(idxTask)-dimTasks(idxTask-1));
            else
                population{idxTask}(i).rnvec=population{idxTask-1}(i).rnvec(1:dimTasks(idxTask));
            end
        end
        for i = nTransfer+1:popSize
            population{idxTask}(i) = Chromosome();
            population{idxTask}(i) = initialize(population{idxTask}(i),dimTasks(idxTask)); % initialize the rest 80% population randomly
        end
        for i = 1 : popSize
            [population{idxTask}(i),callsPerIndividual(i)] = evaluate_SOO(population{idxTask}(i),tasks(idxTask),pIL,options);
        end
        
        totalEvals(r,gen)=totalEvals(r,gen)+sum(callsPerIndividual);
        fCosts=[population{idxTask}.factorial_costs];
        [fCosts,idsCost]=sort(fCosts);
        population{idxTask}=population{idxTask}(idsCost); % sort the chromosomes according to the cost
        bestFitness(r,gen,idxTask)=fCosts(1);
    end
    
    for gen=2:nGen
        totalEvals(r,gen)=totalEvals(r,gen-1);
        for idxTask=1:nTasks
            prevTask=idxTask-1;
            if idxTask==1; prevTask=nTasks; end
            
            % Transfer some chromosomes for the previous task for reproduction
            nTransfer=round(scale(idxTask)*nTransfer0);
            mPrev=mean(reshape([population{prevTask}(1:nTransfer).rnvec],dimTasks(prevTask),nTransfer),2)';
            mThis=mean(reshape([population{idxTask}(1:nTransfer).rnvec],dimTasks(idxTask),nTransfer),2)';
            tempPopulation=population{idxTask}(end:-1:1);
            % replace bad chromosomes in the current population by good
            % chromosomes from the previous population
            if dimTasks(idxTask)>dimTasks(prevTask) % the previous task has a smaller dimensionality
                for i=1:nTransfer
                    tempPopulation(i).rnvec(1:dimTasks(prevTask))=population{prevTask}(i).rnvec(1:dimTasks(prevTask));
                end
                %if gen<600
                    tempPopulation(nTransfer).rnvec(1:dimTasks(prevTask))=population{prevTask}(1).rnvec(1:dimTasks(prevTask))+mThis(1:dimTasks(prevTask))-mPrev(1:dimTasks(prevTask));
                %end
            else
                for i=1:nTransfer
                    tempPopulation(i).rnvec=population{prevTask}(i).rnvec(1:dimTasks(idxTask));
                end
                %if gen<600
                    tempPopulation(nTransfer).rnvec=population{prevTask}(1).rnvec(1:dimTasks(idxTask))+mThis(1:dimTasks(idxTask))-mPrev(1:dimTasks(idxTask));
                %end
            end
            
            idsOrder = [1:nTransfer nTransfer+randperm(popSize-nTransfer)];
            for i = 1 : popSize/2
                p1 = idsOrder(i);
                p2 = idsOrder(i+popSize/2);
                u = rand(1,dimTasks(idxTask));
                cf = zeros(1,dimTasks(idxTask));
                cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                child(2*i-1)=Chromosome();
                child(2*i)=Chromosome();
                child(2*i-1) = crossover(child(2*i-1),tempPopulation(p1),tempPopulation(p2),cf);
                child(2*i) = crossover(child(2*i),tempPopulation(p2),tempPopulation(p1),cf);
                child(2*i-1)=mutate(child(2*i-1),child(2*i-1),dimTasks(idxTask),mum);
                child(2*i)=mutate(child(2*i),child(2*i),dimTasks(idxTask),mum);
                % variable swap
                swap_indicator = (rand(1,dimTasks(idxTask)) >= 0.5);
                temp = child(2*i-1).rnvec(swap_indicator);
                child(2*i-1).rnvec(swap_indicator) = child(2*i).rnvec(swap_indicator);
                child(2*i).rnvec(swap_indicator) = temp;
            end
            
            for i = 1 : popSize
                [child(i),callsPerIndividual(i)] = evaluate_SOO(child(i),tasks(idxTask),pIL,options);
            end
            totalEvals(r,gen)=totalEvals(r,gen)+sum(callsPerIndividual);
            intpopulation(1:popSize)=child;
            intpopulation(popSize+1:2*popSize)=population{idxTask};
            % remove the duplicates in intpopulation
            [rnvec,idsUnique]=unique(reshape([intpopulation.rnvec],dimTasks(idxTask),2*popSize)','rows');
            intpopulation=intpopulation(idsUnique);
            [~,idsCost]=sort([intpopulation.factorial_costs]);
            intpopulation=intpopulation(idsCost); rnvec=rnvec(idsCost,:);
            bestFitness(r,gen,idxTask)=bestFitness(r,gen-1,idxTask);
            if intpopulation(1).factorial_costs<=bestFitness(r,gen,idxTask)
                bestFitness(r,gen,idxTask)=intpopulation(1).factorial_costs;
                bestChromosome(r,idxTask)=intpopulation(1);
            end
            
            if strcmp(selPocess,'elitist')
                if length(intpopulation)>=popSize
                    population{idxTask}=intpopulation(1:popSize);
                else
                    population{idxTask}(1:length(intpopulation))=intpopulation;
                    for i=length(intpopulation)+1:popSize
                        population{idxTask}(i) = Chromosome();
                        population{idxTask}(i) = initialize(population{idxTask}(i),dimTasks(idxTask));
                    end
                end
            elseif strcmp(selPocess,'roulette wheel')
                for i = 1:length(intpopulation)
                    intpopulation(i).scalar_fitness=1/i;
                end
                for i=1:popSize
                    population{idxTask}(i)=intpopulation(RouletteWheelSelection([intpopulation.scalar_fitness]));
                end
            end
            
            % Update scale by rank
            [~,ia]=intersect(rnvec,reshape([child(1:2:2*nTransfer-1).rnvec],dimTasks(idxTask),nTransfer)','rows');
            mRank=mean(ia);
            scale(idxTask)=min(1.5,max(.6,(.3-(mRank/length(intpopulation)-.25*pTransfer))/.2));
        end
    end
end
dataMTEA.wallClockTime=toc;
dataMTEA.bestFitness=bestFitness;
dataMTEA.bestIndData=bestChromosome;
dataMTEA.totalEvals=totalEvals;