function dataEBSGA = EBSGA(tasks,popSize,nGen,selPocess,rmp,pIL,nRepeat,idxTask,dq,initPop)
% EBSGA function: implementation of "Evolutionary Many-tasking Based on Biocoenosis through Symbiosis: A Framework and Benchmark Problems"
% Xianfeng Tan, 05/29/2018, xianfeng_tan@hust.edu.cn
tic;

dataDisp=cell(1,3);
dataDisp{1}=idxTask; dataDisp{2}='EBSGA';

mu = 2;     % Index of Simulated Binary Crossover (tunable)
mum = 5;    % Index of polynomial mutation
nTasks=length(tasks);
while mod(popSize,nTasks)
    popSize = popSize + 1;
end

 dimTasks=zeros(1,nTasks); population=cell(1,nTasks);child=cell(1,nTasks);
for i=1:nTasks
    dimTasks(i) = tasks(i).dims;
end
D_multitask=max(dimTasks);

callsPerIndividual=zeros(1,popSize);
bestFitness = zeros(nRepeat,nGen,nTasks);    % best fitness found
totalEvals=zeros(nRepeat,nGen);   % total number of task evaluations so far
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning

for r = 1:nRepeat
    dataDisp{3}=r;
    dq.send(dataDisp);
    
    % Initialize the first generation of all Task  randomly
    EvalsO=zeros(1,nTasks);
    improveS=zeros(1,nTasks);
    improveO=zeros(1,nTasks);
    RMP=rmp*ones(1,nTasks);
    gen=1;
    for idxTask=1:nTasks
        for i = 1 : popSize
            population{idxTask}(i) = Chromosome();
            population{idxTask}(i) = initialize(population{idxTask}(i),D_multitask);
            if nargin>=10
                population{idxTask}(i).rnvec(1:dimTasks(idxTask))=initPop{idxTask,r}(i,1:dimTasks(idxTask));
            else
                initPop{idxTask,r}(i)=population{idxTask}(i).rnvec;
            end
            [population{idxTask}(i),callsPerIndividual(i)] = evaluate_SOO(population{idxTask}(i),tasks(idxTask),pIL,options);
            population{idxTask}(i).skill_factor=idxTask;
        end
        totalEvals(r,gen)=totalEvals(r,gen)+sum(callsPerIndividual);
        fCosts=[population{idxTask}.factorial_costs];
        [fCosts,idsCost]=sort(fCosts);
        population{idxTask}=population{idxTask}(idsCost); % sort the chromosomes according to their costs
        bestFitness(r,gen,idxTask)=fCosts(1);
    end
    
    for gen=2:nGen
        totalEvals(r,gen)=totalEvals(r,gen-1);
        % Concatenate Offspring
        for idxTask=1:nTasks
            idsOrder = randperm(popSize);
            count=1;
            for i = 1 : popSize/2
                p1 = idsOrder(i);
                p2 = idsOrder(i+popSize/2);
                u = rand(1,D_multitask);
                cf = zeros(1,D_multitask);
                cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                child{idxTask}(count)=Chromosome();
                child{idxTask}(count+1)=Chromosome();
                child{idxTask}(count) = crossover(child{idxTask}(count),population{idxTask}(p1),population{idxTask}(p2),cf);
                child{idxTask}(count+1) = crossover(child{idxTask}(count+1),population{idxTask}(p2),population{idxTask}(p1),cf);
                child{idxTask}(count)=mutate(child{idxTask}(count),child{idxTask}(count),D_multitask,mum);
                child{idxTask}(count+1)=mutate(child{idxTask}(count+1),child{idxTask}(count+1),D_multitask,mum);
                child{idxTask}(count).skill_factor=idxTask;
                child{idxTask}(count+1).skill_factor=idxTask;
                count=count+2;
            end
        end
        
        for idxTask=1:nTasks
            if rand(1)<RMP(idxTask)
                idt=randi(2,popSize,1);
                idsOrder1 = randperm(popSize);
                idsOrder2 = randperm(popSize);
                for i = 1 : popSize
                    if idt(i)==1
                        Candidate(i)=child{1}(idsOrder1(i)) ;
                    else
                        Candidate(i)=child{2}(idsOrder2(i)) ;
                    end
                end
                EvalsO(idxTask)=EvalsO(idxTask)+length(find(idt~=idxTask));
            else
                Candidate=child{idxTask};
            end
            for i = 1 : popSize
                [Candidate(i),callsPerIndividual(i)] = evaluate_SOO(Candidate(i),tasks(idxTask),pIL,options);
            end
            totalEvals(r,gen)=totalEvals(r,gen)+sum(callsPerIndividual);
            intpopulation(1:popSize)=Candidate;
            intpopulation(popSize+1:2*popSize)=population{idxTask};
            [~,idsCost]=sort([intpopulation.factorial_costs]);
            intpopulation=intpopulation(idsCost);
            bestFitness(r,gen,idxTask)=bestFitness(r,gen-1,idxTask);
            if intpopulation(1).factorial_costs<=bestFitness(r,gen,idxTask)
                bestFitness(r,gen,idxTask)=intpopulation(1).factorial_costs;
                bestChromosome(r,idxTask)=intpopulation(1);
                if intpopulation(1).skill_factor==idxTask
                    improveS(idxTask)=improveS(idxTask)+1;
                else
                    improveO(idxTask)=improveO(idxTask)+1;
                end
            end
            
            if strcmp(selPocess,'elitist')
                population{idxTask}=intpopulation(1:popSize);
            elseif strcmp(selPocess,'roulette wheel')
                for i = 1:length(intpopulation)
                    intpopulation(i).scalar_fitness=1/i;
                end
                for i=1:popSize
                    population{idxTask}(i)=intpopulation(RouletteWheelSelection([intpopulation.scalar_fitness]));
                end
            end
            % Update the probability of information exchange
            RO=improveO(idxTask)/EvalsO(idxTask);
            RS=improveS(idxTask)/(totalEvals(r,gen-1)/nTasks+sum(callsPerIndividual)-EvalsO(idxTask));
            RMP(idxTask)=RO/(RO+RS)  ;
        end
    end
end
dataEBSGA.wallClockTime=toc;
dataEBSGA.bestFitness=bestFitness;
dataEBSGA.bestIndData=bestChromosome;
dataEBSGA.totalEvals=totalEvals;