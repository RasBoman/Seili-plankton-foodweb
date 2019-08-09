# Seili-plankton-foodweb
Dynamic Bayes net model with Matlab for Pro Gradu

Current file is meant to be working with 14 variables. 

When tried to run, error message:

>> Run
Unable to perform assignment because the size of the left side is 14-by-14 and the size of the right side is 14-by-14-by-14.

Error in mk_dbn (line 40)
dag(1:n,(1:n)+n) = bnet.inter;

Error in simpleSeili (line 115)
bnet = mk_dbn(intra, inter, ns, 'observed', onodes, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);
