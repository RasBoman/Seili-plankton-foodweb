% Dynamic Bayesian Network for phytoplankton - zooplankton
% dynamics in Archipelago sea. 

% Adapted from Bayesian network originally created by Laura Uusitalo

% Rasmus Boman 2019
% rasmus.a.boman@gmail.com

%%%%%%%%%%%%%%%%%%%%%%%%%%
% This model is the original that
% - uses all the data
% - includes 2 hidden variables (1 generic, 1 for zooplankton)
% - Simplified version of relationships (not differing ciliates)

N = 14; % Number of nodes in the model

% Naming the variables for clarity
DON = 1; DOP = 2; Sal = 3; Temp = 4; HVGen = 5; % Environmental & general HV
Ochro = 6; Hapto = 7; Dino = 8; % Phytoplankton 1/2
Chloro = 9; Cyano = 10; % Phytoplankton 2/2 low nutritional value
HVZoo = 11; % Detritus and hidden variable for zoo
CopeSp = 12; CladoSp = 13; PseudoT = 14; % Zooplanktonic species

% DAG Structure

% "intras" are for one time slice

ss = 14;
intra = zeros(N); % using boolean (true/false) to save computational work

% Environmental variables 
intra(DON, [Ochro, Hapto, Dino, Chloro, Cyano]) = true; %  Dissolved organic nitrogen
intra(DOP, [Ochro, Hapto, Dino, Chloro, Cyano]) = true; % Dissolved organic phosphorus
intra(Sal, [Ochro, Hapto, Dino, Chloro, Cyano, CopeSp, CladoSp, PseudoT]) = true; % salinity
intra(Temp, [Ochro, Hapto, Dino, Chloro, Cyano, CopeSp, CladoSp, PseudoT]) = true; % temperature
intra(HVGen, [Ochro, Hapto, Dino, Chloro, Cyano, CopeSp, CladoSp, PseudoT]) = true; % Generic HV

% Phytoplankton layer
intra(Ochro, [CopeSp, CladoSp, PseudoT]) = true; % Ochrophyta
intra(Hapto, [CopeSp, CladoSp, PseudoT]) = true; % Haptophyta
intra(Dino, [CopeSp, CladoSp, PseudoT]) = true; % Chryptophyta
intra(Chloro, [CopeSp, CladoSp, PseudoT]) = true; % Chlorophyta
intra(Cyano, [CopeSp, CladoSp, PseudoT]) = true; % Cyanophyta
intra(HVZoo, [CopeSp, CladoSp, PseudoT]) = true; % Hidden variable for zooplankton

%%% inter-dependencies %%%

% "inter" refers to the dependencies between time slices

inter = zeros(N); % table to build in the dependecies
inter(HVGen, HVGen) = true; % Hidden variable linked to itself

% temperature predicting next slice as well..
inter(Temp, [Ochro, Hapto, Dino, Chloro, Cyano], [CopeSp, CladoSp, PseudoT]) = true;

% Phytoplankton connections, previous stocks' effect on the next:
inter(Ochro, Ochro) = true; 
inter(Hapto, Hapto) = true;
inter(Dino, Dino) = true;
inter(Chloro, Chloro) = true;
inter(Cyano, Cyano) = true;

% Zooplankton's effect on itself
inter(PseudoT, PseudoT) = true;
inter(CopeSp, CopeSp) = true;
inter(CladoSp, CladoSp) = true;

% Can you do it like this, does the matlab rotate the lists correctly?
% Phyto from TS1 on zooplankton in TS2 (Bottom-up)
% Cyano- and chlorobacteria not very good nutrition
inter(Ochro, [CopeSp, CladoSp, PseudoT]) = true; 
inter(Hapto, [CopeSp, CladoSp, PseudoT]) = true; 
inter(Dino, [CopeSp, CladoSp, PseudoT]) = true; 

% Read in the data
% Missing values encoded as NaN, converted to empty cell
% The file needs to have the variables in the numbered order in columns!!
% Also HVs

data = readmatrix('Seili_matlab_01.csv'); %Jätetään colnames pois
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
onodes = [2:5, 7:8]; 
dnodes = [ ]; % no discrete nodes
ns = ones(1,N);

% Define equivalence classes for the model variables:
%Equivalence classes are needed in order to learn the conditional
%probability tables from the data so that all data related to a variable,
%i.e. data from all years, is used to learn the distribution; the eclass
%specifies which variables "are the same".

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
% log-likelihood), to avoid getting a model that has got stuck to a por local optimum
rng('shuffle') % init the random number generator based on time stamp
bestloglik = -inf; % initialize
for j = 1:10
    j

    % Set the priors N(0,1), with diagonal covariance matrices.
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end

    %Junction tree learning engine for parameter learning
    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 500);
    loglik = LLtrace(length(LLtrace));
    
    %when a better model is found, store it
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
            
    end
end

%save the bestbnet object
save('bestbnet')


t = datlen; % ~30 years

% mean and sd of each of the vars, each time slice
margs=SampleMarg(bestbnet,data(1:datlen,:)',t);

HVGen =[];% Generic HV
HVZoo =[];% Generic HV
     
%write the means and sds of the interest variables down for easier access
for i = 1:t
    i
    %means
    HVGen(i) = margs{1,i}.mu;
    
    %sigma
    HVZoo(i) = margs{1,i}.Sigma;
    end

    %save variables for plotting in R
    save('GenHVMu_onlyGenHV.txt','HVGen','-ascii')
    save('GenHVSig_onlyGenHV.txt','HVZoo','-ascii')
