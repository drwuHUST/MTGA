function dataSOO = SOEA(task,popSize,nGen,selPocess,pIL,nRepeat,idxTask,dq,initPop)
%SOEA function: implementation of SOEA algorithm

tic

dataDisp=cell(1,3);
dataDisp{1}=idxTask; dataDisp{2}='SOEA';

if mod(popSize,2) ~= 0
    popSize = popSize + 1;
end
D = task.dims;
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning

fEvalCalls = zeros(1,nRepeat);
callsPerIndividual=zeros(1,popSize);
bestFitness = zeros(nRepeat,nGen);    % best fitness found
totalEvals=zeros(nRepeat,nGen);   % total number of task evaluations so far
for r = 1:nRepeat
    dataDisp{3}=r;
    dq.send(dataDisp);
    
    for i = 1 : popSize
        population(i) = Chromosome();
        population(i) = initialize(population(i),D);
    end
    if nargin>=9
        for i=1:popSize
            population(i).rnvec=initPop{r}(i);
        end
    else
        initPop{r}=reshape([population.rnvec],D,popSize)';
    end
    for i = 1 : popSize
        [population(i),callsPerIndividual(i)] = evaluate_SOO(population(i),task,pIL,options);
    end
    
    fEvalCalls(r)=fEvalCalls(r) + sum(callsPerIndividual);
    totalEvals(r,1)=fEvalCalls(r);
    bestobj=min([population.factorial_costs]);
    bestFitness(r,1) = bestobj;
    
    generation=1;
    mu = 2;     % Index of Simulated Binary Crossover (tunable)
    mum = 5;    % Index of polynomial mutation
    while generation < nGen
        generation = generation + 1;
        indorder = randperm(popSize);
        count=1;
        for i = 1 : popSize/2
            p1 = indorder(i);
            p2 = indorder(i+(popSize/2));
            child(count)=Chromosome();
            child(count+1)=Chromosome();
            u = rand(1,D);
            cf = zeros(1,D);
            cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
            cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
            child(count) = crossover(child(count),population(p1),population(p2),cf);
            child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
            if rand(1) < 1
                child(count)=mutate(child(count),child(count),D,mum);
                child(count+1)=mutate(child(count+1),child(count+1),D,mum);
            end
            count=count+2;
        end
        for i = 1 : popSize
            [child(i),callsPerIndividual(i)] = evaluate_SOO(child(i),task,pIL,options);
        end
        
        fEvalCalls(r)=fEvalCalls(r) + sum(callsPerIndividual);
        totalEvals(r,generation)=fEvalCalls(r);
        
        intpopulation(1:popSize)=population;
        intpopulation(popSize+1:2*popSize)=child;
        [xxx,y]=sort([intpopulation.factorial_costs]);
        intpopulation=intpopulation(y);
        for i = 1:2*popSize
            intpopulation(i).scalar_fitness=1/i;
        end
        if intpopulation(1).factorial_costs<=bestobj
            bestobj=intpopulation(1).factorial_costs;
            bestIndData(r)=intpopulation(1);
        end
        bestFitness(r,generation)=bestobj;
        
        if strcmp(selPocess,'elitist')
            [xxx,y]=sort(-[intpopulation.scalar_fitness]);
            intpopulation=intpopulation(y);
            population=intpopulation(1:popSize);
        elseif strcmp(selPocess,'roulette wheel')
            for i=1:popSize
                population(i)=intpopulation(RouletteWheelSelection([intpopulation.scalar_fitness]));
            end
        end
        %           disp(['SOO Generation ', num2str(generation), ' best objective = ', num2str(bestobj)])
    end
end
dataSOO.wallClockTime=toc;
dataSOO.bestFitness=bestFitness;
dataSOO.bestIndData=bestIndData;
dataSOO.totalEvals=totalEvals;
dataSOO.initPop=initPop;