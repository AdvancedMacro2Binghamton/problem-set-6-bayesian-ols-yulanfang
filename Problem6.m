clear all; close all; clc;
data = xlsread('data.xlsx');
[n,m] = size(data); 
Y = data(:,1);
X = [ones(n,1) data(:,2:end)];

% 1. OLS
beta = (X'*X)\(X'*Y);
sig_sq = (1/(n-m-1))*(Y-X*beta)'*(Y-X*beta);
v_beta = sig_sq*diag(inv(X'*X));
se_beta = sqrt(v_beta);

k = 0.12;
sig_sq_v = k*[[bsxfun(@times,v_beta,eye(m)) zeros(m,1)];...
    [zeros(1,m) ((2/(n-m))*(sig_sq)^2)]]; 
s = 1e4;
burnin = 1e3;
lag = 1;

% 2a. flat priors;
theta = [beta;sig_sq];
THETA0 = zeros(m+1,s);
acc0_b = [0,0];
acc0 = [0,0];
for i = 1:burnin
       [theta,a] = MH0(theta,sig_sq_v,Y,X);
       acc0_b = acc0_b+[a 1];
end
for i = 1:s
    for j = 1:lag
        [theta,a] = MH0(theta,sig_sq_v,Y,X);
        acc0 = acc0 + [a 1];
    end
    THETA0(:,i) = theta;
    disp(i);
end
acc1_rt = (acc0(1)/acc0(2))*100;
disp(['The acceptance rate is ', num2str(acc1_rt)]);

% Plot posteriors
label = ["\beta_{cons}","\beta_{edu}","\beta_{exp}","\beta_{SMSA}",...
    "\beta_{race}","\beta_{south}","\sigma^2_{\epsilon}"];
for i = 1:m+1
    figure
    histfit(THETA0(i,:),50,'kernel')
    ylabel('Density')
    title(label(:,i))
end

% 2b. beta_edu: normal prior, flat prior for others;
theta = [beta;sig_sq];
THETA1 = zeros(m+1,s);
acc1_b = [0,0];
acc1 = [0,0];
for i = 1:burnin
       [theta,a] = MH1(theta,sig_sq_v,Y,X);
       acc1_b = acc1_b+[a 1];
end
for i = 1:s
    for j = 1:lag
        [theta,a] = MH1(theta,sig_sq_v,Y,X);
        acc1 = acc1 + [a 1];
    end
    THETA1(:,i) = theta;
    disp(i);
end
acc1_rt = (acc1(1)/acc1(2))*100;
disp(['The acceptance rate is ', num2str(acc1_rt)]);

% Plot posteriors
for i = 1:m+1
    figure
    histfit(THETA1(i,:),50,'kernel')
    ylabel('Density')
    title(label(:,i))
end

function [theta_new,a] = MH0(theta_old,sig,Y,X)
theta_p = mvnrnd(theta_old,sig)';
accprob = exp(LogLike0(theta_p,Y,X)-LogLike0(theta_old,Y,X));
u = rand;
if u <= accprob
    theta_new = theta_p; 
    a = 1;
else
    theta_new = theta_old;
    a = 0;
end
end
function [theta_new,a] = MH1(theta_old,sig,Y,X)
theta_p = mvnrnd(theta_old,sig)';
accprob = exp(Loglike1(theta_p,Y,X)-Loglike1(theta_old,Y,X));
u = rand; 
if u <= accprob
    theta_new = theta_p;
    a = 1;
else
    theta_new = theta_old;
    a = 0;
end
end
function logL = LogLike0(theta,Y,X)
[n,m] = size(X); 
Y_hat = X*theta(1:m);
L_1 = zeros(n,1);
for i=1:n
    if theta(m+1) > 0
        L_1(i) = normpdf(Y(i),Y_hat(i),sqrt(theta(m+1)));
    else
        L_1(i) = 0;
    end
end
logL = sum(log(L_1));
end
function logL = Loglike1(theta,Y,X)
[n,m] = size(X); 
beta_edu = theta(2);
beta_edu_prior = normpdf(beta_edu, 0.06, 0.007);
Y_hat = X*theta(1:m);
PY_1 = zeros(n,1);
for i=1:n
    if theta(m+1) > 0
        PY_1(i) = normpdf(Y(i),Y_hat(i),sqrt(theta(m+1)));
    else
        PY_1(i) = 0;
    end
end
logL = log(beta_edu_prior)+sum(log(PY_1));
end
