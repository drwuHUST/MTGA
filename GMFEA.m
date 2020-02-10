function dataGMFEA = GMFEA(tasks,pop,nGen,selectionProcess,rmp,pIL,nRepeat,idxTask,dq,initPop)
% G-MFEA function: implementation of G-MFEA algorithm
tic
dataDisp=cell(1,3);
dataDisp{1}=idxTask; dataDisp{2}='GMFEA';
nTasks=length(tasks);
if nTasks <= 1
    error('At least 2 tasks required for GMFEA');
end
while mod(pop,nTasks) ~= 0
    pop = pop + 1;
end

D=zeros(1,nTasks);
for i=1:nTasks
    D(i)=tasks(i).dims;
end
D_multitask=max(D);
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning

fncevalCalls = zeros(1,nRepeat);
callsPerIndividual=zeros(1,pop);
evBestFitness = zeros(nTasks*nRepeat,nGen);    % best fitness found
TotalEvaluations=zeros(nRepeat,nGen);               % total number of task evaluations so fer
bestobj=Inf(1,nTasks);

for rep = 1:nRepeat
    dataDisp{3}=rep;
    dq.send(dataDisp);
    for i = 1 : pop
        population(i) = Chromosome();
        population(i) = initialize(population(i),D_multitask);
        population(i).skill_factor=0;
    end
    for n=1:nTasks
        if nargin>=10
            for i=1:pop/nTasks
                population((n-1)*pop/nTasks+i).rnvec(1:D(n))=initPop{n,rep}(i,1:D(n));
            end
        else
            initPop{n,rep}=reshape([population((n-1)*pop/nTasks+(1:pop/nTasks)).rnvec],D(n),pop/nTasks)';
        end
    end
    for i = 1 : pop
        [population(i),callsPerIndividual(i)] = evaluate(population(i),tasks,pIL,nTasks,options);
    end
    
    fncevalCalls(rep)=fncevalCalls(rep) + sum(callsPerIndividual);
    TotalEvaluations(rep,1)=fncevalCalls(rep);
    
    factorial_cost=zeros(1,pop);
    for i = 1:nTasks
        for j = 1:pop
            factorial_cost(j)=population(j).factorial_costs(i);
        end
        [xxx,y]=sort(factorial_cost);
        population=population(y);
        for j=1:pop
            population(j).factorial_ranks(i)=j;
        end
        bestobj(i)=population(1).factorial_costs(i);
        evBestFitness(i+2*(rep-1),1)=bestobj(i);
        bestIndData(rep,i)=population(1);
    end
    for i=1:pop
        [xxx,yyy]=min(population(i).factorial_ranks);
        x=find(population(i).factorial_ranks == xxx);
        equivalent_skills=length(x);
        if equivalent_skills>1
            population(i).skill_factor=x(randi(equivalent_skills,1));
            tmp=population(i).factorial_costs(population(i).skill_factor);
            population(i).factorial_costs(1:nTasks)=inf;
            population(i).factorial_costs(population(i).skill_factor)=tmp;
        else
            population(i).skill_factor=yyy;
            tmp=population(i).factorial_costs(population(i).skill_factor);
            population(i).factorial_costs(1:nTasks)=inf;
            population(i).factorial_costs(population(i).skill_factor)=tmp;
        end
    end
    
    mu = 2;     % Index of Simulated Binary Crossover (tunable)
    mum = 5;   % Index of polynomial mutation
    threshold_trans=round(.1*nGen); % The threshold of triggering the variable translation strategy
    frequency_trans=round(.02*nGen); % The frequency of changing the translate direction
    sf=1.25; % The scale factor
    perBest=0.4; % The percentage of the best solutions to estimate current optimums
    cp=0.5*ones(1,D_multitask); % The designated location to which all optimums are transferred to
    generation=1;
    d=zeros(nTasks,D_multitask);
    while generation < nGen
        generation = generation + 1;
        
        tempopulation=population;
        if generation>threshold_trans
            % Update the translated directions of each task
            if mod(generation,frequency_trans)==0
                for i=1:nTasks
                    x=find([population.skill_factor] == i);
                    numBest=round(perBest*length(x));
                    m=mean(reshape([population(x(1:numBest)).rnvec],D_multitask,numBest),2)';
                    d(i,:)=sf*((generation/nGen)^2)*(cp-m);
                end
            end
            % Update the population by the translated directions of each task
            for i=1:pop
                tempopulation(i).rnvec=population(i).rnvec+d(population(i).skill_factor,:);
            end
        end
        
        indorder = randperm(pop);
        count=1;
        for i = 1 : pop/2
            p1 = indorder(i);
            p2 = indorder(i+(pop/2));
            
            % The decision variable shuffling strategy
            D_max=max(D(tempopulation(p1).skill_factor),D(tempopulation(p2).skill_factor));
            Dorder=zeros(nTasks,D_max);
            if D(tempopulation(p1).skill_factor)<D_max
                % randomly select one individual from P that has the same
                % skill factor as p2
                x=find([tempopulation.skill_factor] == tempopulation(p2).skill_factor);
                same_skills=length(x);
                temp=randi(same_skills);
                temp_ind=tempopulation(x(temp)).rnvec;
                % Randomly perturb the order of L1
                Dorder(tempopulation(p1).skill_factor,:) = randperm(D_max);
                temp_ind(Dorder(tempopulation(p1).skill_factor,1:D(tempopulation(p1).skill_factor)))=tempopulation(p1).rnvec(1:D(tempopulation(p1).skill_factor));
                tempopulation(p1).rnvec=temp_ind;
            elseif D(tempopulation(p2).skill_factor)<D_max
                x=find([tempopulation.skill_factor] == tempopulation(p1).skill_factor);
                same_skills=length(x);
                temp=randi(same_skills);
                temp_ind=tempopulation(x(temp)).rnvec;
                Dorder(tempopulation(p2).skill_factor,:) = randperm(D_max);
                temp_ind(Dorder(tempopulation(p2).skill_factor,1:D(tempopulation(p2).skill_factor)))=tempopulation(p2).rnvec(1:D(tempopulation(p2).skill_factor));
                tempopulation(p2).rnvec=temp_ind;
            end
            
            child(count)=Chromosome();
            child(count+1)=Chromosome();
            if (tempopulation(p1).skill_factor == tempopulation(p2).skill_factor) || (rand(1)<rmp)      % crossover
                u = rand(1,D_multitask);
                cf = zeros(1,D_multitask);
                cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                child(count) = crossover(child(count),tempopulation(p1),tempopulation(p2),cf);
                child(count+1) = crossover(child(count+1),tempopulation(p2),tempopulation(p1),cf);
                if rand(1) < 1
                    child(count)=mutate(child(count),child(count),D_multitask,mum);
                    child(count).rnvec=child(count).rnvec-d(tempopulation(p1).skill_factor,:); % Re-transfer the offspring
                    child(count+1)=mutate(child(count+1),child(count+1),D_multitask,mum);
                    child(count+1).rnvec=child(count+1).rnvec-d(tempopulation(p2).skill_factor,:); % Re-transfer the offspring
                end
                sf1=1+round(rand(1));
                sf2=1+round(rand(1));
                if sf1 == 1 % skill factor selection
                    child(count).skill_factor=tempopulation(p1).skill_factor;
                else
                    child(count).skill_factor=tempopulation(p2).skill_factor;
                end
                
                if sf2 == 1
                    child(count+1).skill_factor=tempopulation(p1).skill_factor;
                else
                    child(count+1).skill_factor=tempopulation(p2).skill_factor;
                end
            else
                child(count)=mutate(child(count),tempopulation(p1),D_multitask,mum);
                child(count).rnvec=child(count).rnvec-d(tempopulation(p1).skill_factor,:);
                child(count).skill_factor=tempopulation(p1).skill_factor;
                child(count+1)=mutate(child(count+1),tempopulation(p2),D_multitask,mum);
                child(count+1).rnvec=child(count+1).rnvec-d(tempopulation(p2).skill_factor,:);
                child(count+1).skill_factor=tempopulation(p2).skill_factor;
            end
            % Re-change the order of decision variables
            if D(child(count).skill_factor)<D_max
                child(count).rnvec(1:D(child(count).skill_factor))=child(count).rnvec(Dorder(child(count).skill_factor,1:D(child(count).skill_factor)));
                child(count).rnvec(D(child(count).skill_factor)+1:D_max)=tempopulation(p1).rnvec(D(child(count).skill_factor)+1:D_max);
            end
            if D(child(count+1).skill_factor)<D_max
                child(count+1).rnvec(1:D(child(count+1).skill_factor))=child(count+1).rnvec(Dorder(child(count+1).skill_factor,1:D(child(count+1).skill_factor)));
                child(count+1).rnvec(D(child(count+1).skill_factor)+1:D_max)=tempopulation(p2).rnvec(D(child(count+1).skill_factor)+1:D_max);
            end
            child(count).rnvec(child(count).rnvec>1)=1;
            child(count+1).rnvec(child(count+1).rnvec>1)=1;
            child(count).rnvec(child(count).rnvec<0)=0;
            child(count+1).rnvec(child(count+1).rnvec<0)=0;
            
            count=count+2;
        end
        for i = 1 : pop
            [child(i),callsPerIndividual(i)] = evaluate(child(i),tasks,pIL,nTasks,options);
        end
        fncevalCalls(rep)=fncevalCalls(rep) + sum(callsPerIndividual);
        TotalEvaluations(rep,generation)=fncevalCalls(rep);
        
        intpopulation(1:pop)=population;
        intpopulation(pop+1:2*pop)=child;
        factorial_cost=zeros(1,2*pop);
        for i = 1:nTasks
            for j = 1:2*pop
                factorial_cost(j)=intpopulation(j).factorial_costs(i);
            end
            [xxx,y]=sort(factorial_cost);
            intpopulation=intpopulation(y);
            for j=1:2*pop
                intpopulation(j).factorial_ranks(i)=j;
            end
            if intpopulation(1).factorial_costs(i)<=bestobj(i)
                bestobj(i)=intpopulation(1).factorial_costs(i);
                bestIndData(rep,i)=intpopulation(1);
            end
            evBestFitness(i+2*(rep-1),generation)=bestobj(i);
        end
        for i=1:2*pop
            [xxx,yyy]=min(intpopulation(i).factorial_ranks);
            intpopulation(i).skill_factor=yyy;
            intpopulation(i).scalar_fitness=1/xxx;
        end
        
        if strcmp(selectionProcess,'elitist')
            [xxx,y]=sort(-[intpopulation.scalar_fitness]);
            intpopulation=intpopulation(y);
            population=intpopulation(1:pop);
        elseif strcmp(selectionProcess,'roulette wheel')
            for i=1:nTasks
                skillGroup(i).individuals=intpopulation([intpopulation.skill_factor]==i);
            end
            count=0;
            while count<pop
                count=count+1;
                skill=mod(count,nTasks)+1;
                population(count)=skillGroup(skill).individuals(RouletteWheelSelection([skillGroup(skill).individuals.scalar_fitness]));
            end
        end
        %  disp(['MFEA Generation = ', num2str(generation), ' best factorial costs = ', num2str(bestobj)]);
    end
end
dataGMFEA.wallClockTime=toc;
dataGMFEA.bestFitness=evBestFitness;
dataGMFEA.bestIndData=bestIndData;
dataGMFEA.totalEvals=TotalEvaluations;
dataGMFEA.initPop=initPop;
end