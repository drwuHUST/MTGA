function data_MFEA = LDAMFEA(Tasks,pop,gen,selection_process,rmp,p_il,reps,idxTask,dq,initPop)
%LDA_MFEA function: implementation of LDA_MFEA algorithm
%     clc

warning off all;

tic
dataDisp=cell(1,3);
dataDisp{1}=idxTask; dataDisp{2}='LDAMFEA';
nTasks=length(Tasks);
if nTasks <= 1
    error('At least 2 tasks required for MFEA');
end
while mod(pop,nTasks) ~= 0
    pop = pop + 1;
end

D=zeros(1,nTasks);
for i=1:nTasks
    D(i)=Tasks(i).dims;
end
D_multitask=max(D);
options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','MaxIter',2);  % settings for individual learning

fnceval_calls = zeros(1,reps);
calls_per_individual=zeros(1,pop);
EvBestFitness = zeros(nTasks*reps,gen);    % best fitness found
TotalEvaluations=zeros(reps,gen);               % total number of task evaluations so fer
bestobj=Inf(1,nTasks);
for rep = 1:reps
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
    temp_points = zeros(pop,D_multitask);
    temp_skill = zeros(pop,1);
    points_skill = zeros(pop*gen,1);
    
    for i = 1 : pop
        [population(i),calls_per_individual(i)] = evaluate(population(i),Tasks,p_il,nTasks,options);
    end
    
    fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
    TotalEvaluations(rep,1)=fnceval_calls(rep);
    
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
        EvBestFitness(i+2*(rep-1),1)=bestobj(i);
        bestInd_data(rep,i)=population(1);
    end
    for i=1:pop
        [xxx,yyy]=min(population(i).factorial_ranks);
        x=find(population(i).factorial_ranks == xxx);
        equivalent_skills=length(x);
        if equivalent_skills>1
            population(i).skill_factor=x(1+round((equivalent_skills-1)*rand(1)));
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
    
    %for accumulating historic points.
    PA = [];
    PB = [];
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GENERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    while generation < gen
        generation = generation + 1;
        
        %Extract Task specific Data Sets
        for i = 1:nTasks
            subpops(i).data = [];
            f(i).cost = [];
        end
        
        for i = 1:pop
            subpops(population(i).skill_factor).data = [subpops(population(i).skill_factor).data;population(i).rnvec];
            f(population(i).skill_factor).cost =  [f(population(i).skill_factor).cost;population(i).factorial_costs(population(i).skill_factor)];
        end
        
        tempA = [subpops(1).data,f(1).cost];
        % accumulate all historical points of T1  and sort according to
        % factorial cost
        tempA = [PA;tempA];
        tempA = sortrows(tempA,D(1)+1);
        PA = tempA;
        A = tempA(:,1:end-1);     %extract chromosomes except the last column(factorial_costs)
        %store into matrix A
        
        
        tempB = [subpops(2).data,f(2).cost];
        % accumulate all historical points of T2  and sort according to
        % factorial cost
        tempB = [PB;tempB];
        tempB = sortrows(tempB,D(2)+1);
        PB = tempB;
        B = tempB(:,1:end-1);     %extract chromosomes except the last column(factorial_costs)
        %store into matrix B
        
        s_a = size(A,1);
        s_b = size(B,1);
        
        diff = abs(s_a - s_b);
        %same number of rows for both task populations.
        %for matrix mapping
        if s_a < s_b
            %trim b
            B = B(1:end-diff,:);
        else A = A(1:end-diff,:);
        end
        %current row count of each of the populations row (a == b).
        % curr_row1 = size(A,1);
        %curr_row2 = size(B,1);
        
        %find Linear Least square mapping between two tasks.
        
        if (D(1) > D(2))   %Different dimensions : map T2 to T1
            [m1,m2] = mapping(B(:,1:D(2)),A);
            
            
        else
            [m1,m2] = mapping(A,B);  %Same dimensions : map T1 to T2
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Evolution phase: Crossover or LDA + Crossover
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        tempv = Chromosome();
        indorder = randperm(pop);
        count=1;
        for i = 1 : pop/2
            p1 = indorder(i);
            p2 = indorder(i+(pop/2));
            child(count)=Chromosome();
            child(count+1)=Chromosome();
            
            %----------CROSSOVER
            if (population(p1).skill_factor == population(p2).skill_factor || rand(1)< rmp)      % crossover
                u = rand(1,D_multitask);
                cf = zeros(1,D_multitask);
                cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                child(count) = crossover(child(count),population(p1),population(p2),cf);
                child(count+1) = crossover(child(count+1),population(p2),population(p1),cf);
                %                     if rand(1) < 1
                %                         child(count)=mutate(child(count),child(count),D_multitask,mum);
                %                         child(count+1)=mutate(child(count+1),child(count+1),D_multitask,mum);
                %                     end
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
                %%%%%% ----------LDA + CROSSOVER--------------------------
                
                %same dimensions : assuming mapping is always from T1
                %to T2 for D1 = D2.
                if (D(1) == D(2))
                    if (population(p1).skill_factor == 1)
                        
                        tempv.rnvec = population(p1).rnvec *m1;
                        
                        %crossover
                        u = rand(1,D_multitask);
                        cf = zeros(1,D_multitask);
                        cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                        cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                        child(count) = crossover(child(count),tempv,population(p2),cf);
                        child(count+1) = crossover(child(count+1),population(p2),tempv,cf);
                        
                        sf1=1+round(rand(1));
                        sf2=1+round(rand(1));
                        if sf1 == 1 % skill factor selection
                            child(count).skill_factor=population(p1).skill_factor;
                            child(count).rnvec =  child(count).rnvec * m2;
                        else
                            child(count).skill_factor=population(p2).skill_factor;
                        end
                        
                        if sf2 == 1
                            child(count+1).skill_factor=population(p1).skill_factor;
                            child(count+1).rnvec = child(count+1).rnvec * m2;
                        else
                            child(count+1).skill_factor=population(p2).skill_factor;
                        end
                        
                        %else P(2).skill_factor ==1
                    else
                        tempv.rnvec = population(p2).rnvec *m1;
                        
                        %crossover
                        u = rand(1,D_multitask);
                        cf = zeros(1,D_multitask);
                        cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                        cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                        child(count) = crossover(child(count),tempv,population(p1),cf);
                        child(count+1) = crossover(child(count+1),population(p1),tempv,cf);
                        
                        sf1=1+round(rand(1));
                        sf2=1+round(rand(1));
                        if sf1 == 1 % skill factor selection
                            child(count).skill_factor=population(p2).skill_factor;
                            child(count).rnvec = child(count).rnvec * m2;
                        else
                            child(count).skill_factor=population(p1).skill_factor;
                        end
                        
                        if sf2 == 1
                            child(count+1).skill_factor=population(p2).skill_factor;
                            child(count+1).rnvec = child(count+1).rnvec * m2;
                        else
                            child(count+1).skill_factor=population(p1).skill_factor;
                        end
                        
                        
                    end % if population(p1).skill_factor == 1)
                    
                    
                end %if (D(1)==D(2))
                
                
                %different dimensions : map T2 to T1 (Prob 6)
                if (D(1) > D(2))
                    
                    if (population(p1).skill_factor == 1)
                        
                        tempv.rnvec = population(p2).rnvec(1:D(2)) *m1;
                        
                        
                        %crossover
                        u = rand(1,D_multitask);
                        cf = zeros(1,D_multitask);
                        cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                        cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                        child(count) = crossover(child(count),tempv,population(p1),cf);
                        child(count+1) = crossover(child(count+1),population(p1),tempv,cf);
                        
                        sf1=1+round(rand(1));
                        sf2=1+round(rand(1));
                        if sf1 == 1 % skill factor selection
                            child(count).skill_factor=population(p1).skill_factor;
                            
                        else
                            child(count).skill_factor=population(p2).skill_factor;
                            child(count).rnvec(1:D(2)) = child(count).rnvec * m2;
                        end
                        
                        if sf2 == 1
                            child(count+1).skill_factor=population(p1).skill_factor;
                        else
                            child(count+1).skill_factor=population(p2).skill_factor;
                            child(count+1).rnvec(1:D(2)) = child(count+1).rnvec * m2;
                        end
                        
                    else % P(2).skill_factor == 1
                        
                        tempv.rnvec = population(p1).rnvec(1:D(2)) *m1;
                        
                        
                        %crossover
                        u = rand(1,D_multitask);
                        cf = zeros(1,D_multitask);
                        cf(u<=0.5)=(2*u(u<=0.5)).^(1/(mu+1));
                        cf(u>0.5)=(2*(1-u(u>0.5))).^(-1/(mu+1));
                        child(count) = crossover(child(count),tempv,population(p2),cf);
                        child(count+1) = crossover(child(count+1),population(p2),tempv,cf);
                        
                        sf1=1+round(rand(1));
                        sf2=1+round(rand(1));
                        if sf1 == 1 % skill factor selection
                            child(count).skill_factor=population(p2).skill_factor;
                            
                        else
                            child(count).skill_factor=population(p1).skill_factor;
                            child(count).rnvec(1:D(2)) = child(count).rnvec * m2;
                        end
                        
                        if sf2 == 1
                            child(count+1).skill_factor=population(p2).skill_factor;
                            
                        else
                            child(count+1).skill_factor=population(p1).skill_factor;
                            child(count+1).rnvec(1:D(2)) = child(count+1).rnvec * m2;
                        end
                        
                    end
                    
                    
                end  %end if D(1) > D(2)
                
                %
                %
                %                     child(count)=mutate(child(count),population(p1),D_multitask,mum);
                %                     child(count).skill_factor=population(p1).skill_factor;
                %                     child(count+1)=mutate(child(count+1),population(p2),D_multitask,mum);
                %                     child(count+1).skill_factor=population(p2).skill_factor;
                
                
            end
            count=count+2;
        end
        
        for i = 1 : pop
            [child(i),calls_per_individual(i)] = evaluate(child(i),Tasks,p_il,nTasks,options);
        end
        fnceval_calls(rep)=fnceval_calls(rep) + sum(calls_per_individual);
        TotalEvaluations(rep,generation)=fnceval_calls(rep);
        
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
                bestInd_data(rep,i)=intpopulation(1);
            end
            EvBestFitness(i+2*(rep-1),generation)=bestobj(i);
        end
        for i=1:2*pop
            [xxx,yyy]=min(intpopulation(i).factorial_ranks);
            intpopulation(i).skill_factor=yyy;
            intpopulation(i).scalar_fitness=1/xxx;
        end
        
        if strcmp(selection_process,'elitist')
            [xxx,y]=sort(-[intpopulation.scalar_fitness]);
            intpopulation=intpopulation(y);
            population=intpopulation(1:pop);
        elseif strcmp(selection_process,'roulette wheel')
            for i=1:nTasks
                skill_group(i).individuals=intpopulation([intpopulation.skill_factor]==i);
            end
            count=0;
            while count<pop
                count=count+1;
                skill=mod(count,nTasks)+1;
                population(count)=skill_group(skill).individuals(RouletteWheelSelection([skill_group(skill).individuals.scalar_fitness]));
            end
        end
        
        
        %             disp(['MFEA Generation = ', num2str(generation), ' best factorial costs = ', num2str(bestobj)]);
    end %gen
end %rep

data_MFEA.wallClockTime=toc;
data_MFEA.bestFitness=EvBestFitness;
data_MFEA.bestIndData=bestInd_data;
data_MFEA.totalEvals=TotalEvaluations;
data_MFEA.initPop=initPop;
end


function [m1,m2] = mapping(a,b)
m1 = (inv(transpose(a)*a)) * (transpose(a)*b);
m2 = transpose(m1) * (inv(m1*transpose(m1)));
end