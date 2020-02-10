function dataMFEALBS = MFEALBS(tasks,pop,nGen,~,rmp,pIL,nRepeat,idxTask,dq,initPop)
% MFEALBS function: implementation of "Evolutionary Multitasking in Permutation-Based Combinatorial Optimization Problems: Realization with TSP, QAP, LOP, and JSP"
% Xianfeng Tan, 05/29/2018, xianfeng_tan@hust.edu.cn
tic
dataDisp=cell(1,3);
dataDisp{1}=idxTask; dataDisp{2}='MFEALBS';
nTasks=length(tasks);
if nTasks <= 1
    error('At least 2 tasks required for MFEALBS');
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
            initPop{n,rep}=reshape([population((n-1)*pop/nTasks+(1:pop/nTasks)).rnvec],D_multitask,pop/nTasks)';
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
    mum = 5;    % Index of polynomial mutation
    generation=1;
    while generation < nGen
        generation = generation + 1;
        indorder = randperm(pop);
        count=1;
        for i = 1 : pop/2
            p1 = indorder(i);
            p2 = indorder(i+(pop/2));
            child(count)=Chromosome();
            child(count+1)=Chromosome();
            if (population(p1).skill_factor == population(p2).skill_factor) || (rand(1)<rmp)      % crossover
                u = rand(1,D_multitask);
                cf = zeros(1,D_multitask);
                cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                child(count) = crossover(child(count),population(p1),population(p2),cf);
                child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
                if rand(1) < 1
                    child(count)=mutate(child(count),child(count),D_multitask,mum);
                    child(count+1)=mutate(child(count+1),child(count+1),D_multitask,mum);
                end
                sf1=1+round(rand(1));
                sf2=1+round(rand(1));
                if sf1 == 1 % skill factor selection
                    child(count).skill_factor=population(p1).skill_factor;
                else
                    child(count).skill_factor=population(p2).skill_factor;
                end
                
                if sf2 == 1
                    child(count+1).skill_factor=population(p1).skill_factor;
                else
                    child(count+1).skill_factor=population(p2).skill_factor;
                end
                
            else
                child(count)=mutate(child(count),population(p1),D_multitask,mum);
                child(count).skill_factor=population(p1).skill_factor;
                child(count+1)=mutate(child(count+1),population(p2),D_multitask,mum);
                child(count+1).skill_factor=population(p2).skill_factor;
            end
            count=count+2;
        end
        for i = 1 : pop
            [child(i),callsPerIndividual(i)] = evaluate(child(i),tasks,pIL,nTasks,options);
        end
        fncevalCalls(rep)=fncevalCalls(rep) + sum(callsPerIndividual);
        TotalEvaluations(rep,generation)=fncevalCalls(rep);
        
        intpopulation(1:pop)=population;
        intpopulation(pop+1:2*pop)=child;
        for i = 1:nTasks
            x=find([intpopulation.skill_factor] == i);
            same_skills=length(x);
            factorial_cost=nan(1,same_skills);
            for j = 1:same_skills
                factorial_cost(j)=intpopulation(x(j)).factorial_costs(i);
            end
            [~,y]=sort(factorial_cost);
            x=x(y);
            if i==1
                L1=x;
            else
                L2=x;
            end
            if intpopulation(x(1)).factorial_costs(i)<=bestobj(i)
                bestobj(i)=intpopulation(x(1)).factorial_costs(i);
                bestIndData(rep,i)=intpopulation(x(1));
            end
            evBestFitness(i+2*(rep-1),generation)=bestobj(i);
        end
        % Level-Based Selection
        if same_skills>=pop/2 && same_skills<=3*pop/2
            population(1:pop/2)=intpopulation(L1(1:pop/2));
            population(1+pop/2:pop)=intpopulation(L2(1:pop/2));
        elseif same_skills<pop/2
            population(1+same_skills:pop)=intpopulation(L1(1:pop-same_skills));
            population(1:same_skills)=intpopulation(L2(1:same_skills));
        else
            population(1+same_skills-pop:pop)=intpopulation(L1(1:2*pop-same_skills));
            population(1:same_skills-pop)=intpopulation(L2(1:same_skills-pop));
        end
    end
end
dataMFEALBS.wallClockTime=toc;
dataMFEALBS.bestFitness=evBestFitness;
dataMFEALBS.bestIndData=bestIndData;
dataMFEALBS.totalEvals=TotalEvaluations;
dataMFEALBS.initPop=initPop;
end