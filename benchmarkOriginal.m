function [Tasks, g1, g2] = benchmark(index)
%BENCHMARK function
%   Input
%   - index: the index number of problem set
%
%   Output:
%   - Tasks: benchmark problem set
%   - g1: global optima of Task 1
%   - g2: global optima of Task 2
switch(index)
    case 1 % complete intersection with high similarity, Griewank and Rastrigin
        load('Tasks\CI_H.mat')  % loading data from folder .\Tasks
        dim = 50;
        Tasks(1).dims = dim;    % dimensionality of Task 1
        Tasks(1).fnc = @(x)Griewank(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-100*ones(1,dim);   % Upper bound of Task 1
        Tasks(1).Ub=100*ones(1,dim);    % Lower bound of Task 1
        
        Tasks(2).dims = dim;    % dimensionality of Task 2
        Tasks(2).fnc = @(x)Rastrigin(x,Rotation_Task2,GO_Task2);
        Tasks(2).Lb=-50*ones(1,dim);    % Upper bound of Task 2
        Tasks(2).Ub=50*ones(1,dim);     % Lower bound of Task 2
        
        g1 = GO_Task1;
        g2 = GO_Task2;
    case 2 % complete intersection with medium similarity, Ackley and Rastrigin
        load('Tasks\CI_M.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Ackley(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-50*ones(1,dim);
        Tasks(1).Ub=50*ones(1,dim);
        
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Rastrigin(x,Rotation_Task2,GO_Task2);
        Tasks(2).Lb=-50*ones(1,dim);
        Tasks(2).Ub=50*ones(1,dim);
        
        g1 = GO_Task1;
        g2 = GO_Task2;
    case 3 % complete intersection with low similarity, Ackley and Schwefel
        load('Tasks\CI_L.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Ackley(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-50*ones(1,dim);
        Tasks(1).Ub=50*ones(1,dim);
        
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Schwefel(x);
        Tasks(2).Lb=-500*ones(1,dim);
        Tasks(2).Ub=500*ones(1,dim);
        
        g1 = GO_Task1;
        g2 = 420.9687*ones(1,dim);
    case 4 % partially intersection with high similarity, Rastrigin and Sphere
        load('Tasks\PI_H.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Rastrigin(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-50*ones(1,dim);
        Tasks(1).Ub=50*ones(1,dim);
        
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Sphere(x,GO_Task2);
        Tasks(2).Lb=-100*ones(1,dim);
        Tasks(2).Ub=100*ones(1,dim);
        
        g1 = GO_Task1;
        g2 = GO_Task2;
    case 5 % partially intersection with medium similarity, Ackley and Rosenbrock
        load('Tasks\PI_M.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Ackley(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-50*ones(1,dim);
        Tasks(1).Ub=50*ones(1,dim);
        
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Rosenbrock(x);
        Tasks(2).Lb=-50*ones(1,dim);
        Tasks(2).Ub=50*ones(1,dim);
        
        g1 = GO_Task1;
        g2 = ones(1,dim);
    case 6 % partially intersection with low similarity, Ackley and Weierstrass
        load('Tasks\PI_L.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Ackley(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-50*ones(1,dim);
        Tasks(1).Ub=50*ones(1,dim);
        
        dim = 25;
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Weierstrass(x,Rotation_Task2,GO_Task2);
        Tasks(2).Lb=-0.5*ones(1,dim);
        Tasks(2).Ub=0.5*ones(1,dim);
        
        g1 = GO_Task1;
        g2 = GO_Task2;
    case 7 % no intersection with high similarity, Rosenbrock and Rastrigin
        load('Tasks\NI_H.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Rosenbrock(x);
        Tasks(1).Lb=-50*ones(1,dim);
        Tasks(1).Ub=50*ones(1,dim);
        
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Rastrigin(x,Rotation_Task2,GO_Task2);
        Tasks(2).Lb=-50*ones(1,dim);
        Tasks(2).Ub=50*ones(1,dim);
        
        g1 = ones(1,dim);
        g2 = GO_Task2;
    case 8 % no intersection with medium similarity, Griewank and Weierstrass
        load('Tasks\NI_M.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Griewank(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-100*ones(1,dim);
        Tasks(1).Ub=100*ones(1,dim);
        
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Weierstrass(x,Rotation_Task2,GO_Task2);
        Tasks(2).Lb=-0.5*ones(1,dim);
        Tasks(2).Ub=0.5*ones(1,dim);
        
        g1 = GO_Task1;
        g2 = GO_Task2;
    case 9 % no overlap with low similarity, Rastrigin and Schwefel
        load('Tasks\NI_L.mat')
        dim = 50;
        Tasks(1).dims = dim;
        Tasks(1).fnc = @(x)Rastrigin(x,Rotation_Task1,GO_Task1);
        Tasks(1).Lb=-50*ones(1,dim);
        Tasks(1).Ub=50*ones(1,dim);
        
        Tasks(2).dims = dim;
        Tasks(2).fnc = @(x)Schwefel(x);
        Tasks(2).Lb=-500*ones(1,dim);
        Tasks(2).Ub=500*ones(1,dim);
        
        g1 = GO_Task1;
        g2 = 420.9687*ones(1,dim);
end
end