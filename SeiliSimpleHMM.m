% Dynamic Bayesian Network for phytoplankton - zooplankton
% dynamics in Archipelago sea. 

% Adapted from Bayesian network originally created by Laura Uusitalo

% Rasmus Boman 2019
% rasmus.a.boman@gmail.com

%%%%%%%%%%%%%%%%%%%%%%%%%%
% This model is the original that includes
% - 4 environmental variables
% - phytoplankton in 7 variables (class)
% - zooplankton in 7 variables (up to species level)
% - includes 2 hidden variables (1 generic, 1 for zooplankton [NA])

% - Only hidden variable linked to itself between timeslices!!

% Original variables in R:
% [[1] "month"            "dis_org_nitr"     "dis_org_pho"      "sal"              "temp"             "hvgen"           
% [7] "Diatomophyceae"   "Dinophyceae"      "Litostomatea"     "Cyanophyceae"     "Cryptophyceae"    "Chrysophyceae"   
% [13] "Prymnesiophyceae" "hvzoo"            "AcartiaTot"       "DaphniaTot"       "Eubosmina_long"   "Eurytemora_aff"  
% [19] "Evadne_normanni"  "Pleopsis_polyp"   "Synchaeta_sp"    

N = 21; % Number of nodes in the model

% Naming the variables for clarity
Quarter = 1; % Month as discrete variable

DON = 2; DOP = 3; Sal = 4; Temp = 5; HVGen = 6; % Environmental & general HV

Diatom = 7; Dino = 8; Lito = 9; Cyano = 10; % Phytoplankton 1/2
Crypto = 11; Chryso = 12; Prymne = 13; % Phytoplankton 2/2 low nutritional value

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
%intra([7:9 11:13] , 15:20) = true; % Phyto (except Cyanoph.) -> zooplankton
intra(14, 15:21) = true; % Hidden variable -> zooplankton

% time variable
% intra(1, [2:5 7:13 15:21]) = true; % Quarter as a general explanatory variable / categorical

%%% inter-dependencies %%%

% "inter" refers to the dependencies between time slices

inter = zeros(N); % table to build in the dependecies
inter(HVGen, HVGen) = true; % Hidden variable linked to itself
inter(HVZoo, HVZoo) = true;

% temperature predicting next slice as well:
%inter(Temp, [7:13 15:21]) = true;

% Phytoplankton connections, previous stocks' effect on the next:
%inter(Diatom, Diatom) = true; 
%inter(Dino, Dino) = true;
%inter(Lito, Lito) = true;
%inter(Cyano, Cyano) = true;
%inter(Crypto, Crypto) = true;
%inter(Chryso, Chryso) = true;
%inter(Prymne, Prymne) = true;

% Zooplankton's effect on itself
%inter(Acartia, Acartia) = true;
%inter(Daphnia, Daphnia) = true;
%inter(Eubos, Eubos) = true;
%inter(Euryt, Euryt) = true;
%inter(Evadne, Evadne) = true;
%inter(Pleopsis, Pleopsis) = true;
%inter(Synch, Synch) = true;

% Phytoplankton from TS1 -> zooplankton TS2 (Bottom-up)
% Not including cyanobacteria and synchaeta
%inter([7:9 11:13] , 15:20) = true ; 

% Read in the data
% Missing values encoded as NaN, converted to empty cell
% The file needs to have the variables in the numbered order in columns!!
% Also HVs

data = readmatrix('seili_by_quarter_log_scaled_sliced.csv'); 
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
% Hidden variables will not be observed (variables 5 and 13)

onodes = [2:5, 7:13, 15:21]; % Quarter as a continuous node as well..
dnodes = [ ]; % Quarter is not a discrete node
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

for j = 1:80  % Just five times to test / Rasmus 10.08.2019
    j

    % Set the priors N(0,1), with diagonal covariance matrices.
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end

    % Junction tree learning engine for parameter learning
    
    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 500); %changed to 50 for testing. / Rasmus 09.08.2019
    loglik = LLtrace(length(LLtrace));
    
    %when a better model is found, store it
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
            
    end
end

%save the bestbnet object
save('bestbnet_by_class')


t = datlen; % ~30 years

% mean and sd of each of the vars, each time slice
margs=SampleMarg(bestbnet,data(1:datlen,:)',t);

HVGenMu =[];% Generic HV
HVZooMu =[];% Zooplankton HV

HVGenSigma = []; %Generic HV 
HVZooSigma = []; %ZooplanktonHV
     
%write the means (Mu) and sds(sigma) of the interest variables down for easier access
for i = 1:t
    i
    %means
    HVGenMu(i) = margs{6,i}.mu;
    HVZooMu(i) = margs{14,i}.mu;
    
    %sigma
    HVGenSigma(i) = margs{6,i}.Sigma;
    HVZooSigma(i) = margs{14,i}.Sigma;
end

    %save variables for plotting in R
    save('Seili_non_dynamic_just_HVGenMu_VERSIOB.txt','HVGenMu','-ascii')
    save('Seili_non_dynamic_just_HVZooMu_VERSIOB.txt','HVZooMu','-ascii')
    
    save('Seili_non_dynamic_just_HVGenSigma_VERSIOB.txt','HVGenSigma','-ascii')
    save('Seili_non_dynamic_just_HVZooSigma_VERSIOB.txt','HVZooSigma','-ascii')