% Dynamic Bayesian Network for phytoplankton - zooplankton
% dynamics in Archipelago sea. 

% Adapted from Bayesian network originally created by Laura Uusitalo

% Rasmus Boman 2019
% rasmus.a.boman@gmail.com

%%%%%%%%%%%%%%%%%%%%%%%%%%
% This model is the original that

% - phytoplankton in 7 variables (sorted by ~class)
% - zooplankton in 7 variables (up to species level)
% - includes 2 hidden variables (1 generic, 1 for zooplankton, [NA]s in R)

% - Linked timewise
% - Doesn't take quarter into account, must be considered in analysis

% Original variables in R:
% [[1] "month"            "dis_org_nitr"     "dis_org_pho"      "sal"              "temp"             "hvgen"           
% [7] "Diatomophyceae"   "Dinophyceae"      "Litostomatea"     "Cyanophyceae"     "Cryptophyceae"    "Chrysophyceae"   
% [13] "Prymnesiophyceae" "hvzoo"            "AcartiaTot"       "DaphniaTot"       "Eubosmina_long"   "Eurytemora_aff"  
% [19] "Evadne_normanni"  "Pleopsis_polyp"   "Synchaeta_sp"    

N = 21; % Number of nodes in the model

% Naming the variables for clarity

Month = 1; % Month as discrete variable

DON = 2; DOP = 3; Sal = 4; Temp = 5; HVGen = 6; % Environmental & general HV

Diatom = 7; Dino = 8; Lito = 9; Cyano = 10; % Phytoplankton 1/2
Crypto = 11; Chryso = 12; Prymne = 13; % Phytoplankton 2/2 

HVZoo = 14; % Detritus and hidden variable for zoo

Acartia = 15; Daphnia = 16; Eubos = 17; Euryt = 18; % Zooplanktonic species
Evadne = 19; Pleopsis = 20; Synch = 21; 

% DAG Structure

% "intras" are for one time slice

ss = 21;
intra = zeros(N); % using boolean (true/false) to save computational work

% Environmental variables 
intra(DON, 7:13) = true; %  Dissolved organic nitrogen -> phytoplankton
intra(DOP, 7:13) = true; % Dissolved organic phosphorus -> phytoplankton
intra(Sal, [7:13 15:21]) = true; % salinity -> all plankton
intra(Temp, [7:13 15:21]) = true; % temperature -> all plankton 
intra(HVGen, [7:13 15:21]) = true; % Generic HV -> all plankton

% Phytoplankton layer
intra([7:9 11:13] , 15:20) = true; % Phytoplankton (except Cyanoph.) -> zooplankton
intra(14, 15:21) = true; % Hidden variable -> zooplankton

% time variable not taken into account in this model
% intra(1, [7:13 15:21]) = true; % Month as a general explanatory variable / gategorical

%%% inter-dependencies %%%

% "inter" refers to the dependencies between time slices

inter = zeros(N); % table to build in the dependecies
inter(HVGen, HVGen) = true; % Hidden variable linked to itself
inter(HVZoo, HVZoo) = true; % Hidden zooplankton variable linked to itself

% temperature predicting next slice as well:
 inter(Temp, [7:13 15:21]) = true;

% 
% Phytoplankton connections, previous stocks' effect on the next:
inter(Diatom, Diatom) = true; 
inter(Dino, Dino) = true;
inter(Lito, Lito) = true;
inter(Cyano, Cyano) = true;
inter(Crypto, Crypto) = true;
inter(Chryso, Chryso) = true;
inter(Prymne, Prymne) = true;

% Zooplankton's effect on itself
inter(Acartia, Acartia) = true;
inter(Daphnia, Daphnia) = true;
inter(Eubos, Eubos) = true;
inter(Euryt, Euryt) = true;
inter(Evadne, Evadne) = true;
inter(Pleopsis, Pleopsis) = true;
inter(Synch, Synch) = true;

% Phytoplankton from TS1 -> zooplankton TS2 (Bottom-up)
% Not including cyanobacteria and synchaeta
inter([7:9 11:13] , 15:20) = true ; 

% Read in the data
% Missing values encoded as NaN, converted to empty cell
% The file needs to have the variables in the numbered order in columns!!
% Also HVs

data = readmatrix('seili_predict_zoopl.csv'); 
data = num2cell(data);
[datlen, datn] = size(data);
for i = 1:datlen
    for j = 1:datn
        if isnan(data{i, j})
            data{i,j} = [];
        end
    end
end


% Which nodes will be observed? 
% Hidden variables will not be observed (variables 6 and 14)

onodes = [2:5, 7:13, 15:21]; 
dnodes = []; % Month is a discrete node
ns = ones(1,N);

% Define equivalence classes for the model variables:
% Equivalence classes are needed in order to learn the conditional
% probability tables from the data so that all data related to a variable,
% i.e. data from all years, is used to learn the distribution; the eclass
% specifies which variables "are the same".

% In the first year, all vars have their own eclasses;
% in the consecutive years, each variable belongs to the same eclass 
% with itself from the other time slices. 
% This is because due to the temporal dependencies, some of the variables have a
% different number of incoming arcs, and therefore cannot be in the same
% eclass. 

eclass1 = 1:N; % first time slice
eclass2 = (N+1):(2*N);% consecutive time slices
eclass = [eclass1 eclass2];
 
% Make the model
bnet = mk_dbn(intra, inter, ns, 'observed', onodes, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);

%
% Loop over the EM learning 100 times, keep the best model (based on
% log-likelihood), to avoid getting a model that has got stuck to a poor local optimum
rng(13,'twister') % init the random number generator based on time stamp / changed to twister on 09.08.2019 (error) 
bestloglik = -inf; % initialize

for j = 1:2 % 
    j

    % Set the priors N(0,1), with diagonal covariance matrices.
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end

    % Junction tree learning engine for parameter learning
    
    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 5); 
    loglik = LLtrace(length(LLtrace));
    
    %when a better model is found, store it
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
            
    end
end

%save the bestbnet object
save('bestbnet_by_class')

% PREDICTION of zooplankton variables
%

% predict 3 years into the future, use the first 25 years as evidence
t=datlen % The zooplankton is already NA in the final 3 years (3x4 quarters) of the table

% SampleMarg function gives the mean and sd of each of the vars, each time step
margs=SampleMarg(bestbnet, data(1:datlen,:)',t);

% predict the zooplankton variables that have been observed

AcartiaMu = []; 
DaphniaMu = []; 
EubosMu = []; 
EurytMu = []; 
EvadneMu = []; 
PleopsisMu = [];
SynchMu = [];

AcartiaSig = []; 
DaphniaSig = []; 
EubosSig = []; 
EurytSig = []; 
EvadneSig = []; 
PleopsisSig = [];
SynchSig = []; 

% Acartia = 15; Daphnia = 16; Eubos = 17; Euryt = 18; % Zooplanktonic species
% Evadne = 19; Pleopsis = 20; Synch = 21;
%write the means and sds of the interest variables down for easier access
    for i = 80:t
        i
        %means
        AcartiaMu(i) = margs{15,i}.mu;
        DaphniaMu(i) = margs{16,i}.mu;
        EubosMu(i) = margs{17,i}.mu;
        EurytMu(i) = margs{18,i}.mu;
        EvadneMu(i) = margs{19,i}.mu;
        PleopsisMu(i) = margs{20,i}.mu;
        SynchMu(i) = margs{21,i}.mu;
        
        %sds
        AcartiaSig(i) = margs{15,i}.Sigma;
        DaphniaSig(i) = margs{16,i}.Sigma;
        EubosSig(i) = margs{17,i}.Sigma;
        EurytSig(i) = margs{18,i}.Sigma;
        EvadneSig(i) = margs{19,i}.Sigma;
        PleopsisSig(i) = margs{20,i}.Sigma;
        SynchSig(i) = margs{21,i}.Sigma;
        
    end
    
%save variables
save('AcartiaMu_HMM_Pred.txt','AcartiaMu','-ascii')
save('AcartiaSigma_HMM_Pred.txt','AcartiaSig','-ascii')

save('DaphniaMu_HMM_Pred.txt','DaphniaMu','-ascii')
save('DaphniaSigma_HMM_Pred.txt','DaphniaSig','-ascii')

save('EubosMu_HMM_Pred.txt','EubosMu','-ascii')
save('EubosSigma_Pred.txt','EubosSig','-ascii')

save('EurytMu_HMM_Pred.txt','EurytMu','-ascii')
save('EurytSigma_HMM_Pred.txt','EurytSig','-ascii')

save('EvadneMu_HMM_Pred.txt','EvadneMu','-ascii')
save('EvadneSigma_HMM_Pred.txt','EvadneSig','-ascii')

save('PleopsisMu_HMM_Pred.txt','PleopsisMu','-ascii')
save('PleopsisSigma_HMM_Pred.txt','PleopsisSig','-ascii')

save('SynchMu_HMM_Pred.txt','SynchMu','-ascii')
save('SynchSigma_HMM_Pred.txt','SynchSig','-ascii')