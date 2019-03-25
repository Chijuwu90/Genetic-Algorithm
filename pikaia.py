# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:48:04 2019

@author: ChiJuWu
"""
import numpy
import time
import copy

#----------------------------------------------------------------------------
def pikaia_select(npop, jfit, fdif):
    npop1 = npop + 1
    dice = numpy.random.uniform(0, 1)*npop*npop1
    rtfit = 0.
    for i in range(npop):
        rtfit = rtfit + npop1 + fdif*(npop1 - 2*jfit[i])
        if rtfit >= dice:
            idad = i
            return idad
#----------------------------------------------------------------------------
def pikaia_encode(npar, npos, phenotype):
    #take each parameter and turn it into string
    geno = ''
    for i in range(npar):
        temp = phenotype[i]
        geno = geno + str(temp - numpy.floor(temp))[2:2+npos+1]
    return geno
#----------------------------------------------------------------------------
def pikaia_crossover(npar, num_dec, cross_probability, geno1, geno2, seeding):
    length = len(geno1)
    
    numpy.random.seed(seeding)
    rnd = numpy.random.uniform(0, 1)
    
    if rnd < cross_probability:
        
        numpy.random.seed(seeding)
        temp = numpy.random.uniform(0, 1)
        
        split = int(temp*length) + 1
        tmp1a = geno1[0:split+1]
        tmp1b = geno1[split:length]
        tmp2a = geno2[0:split+1]
        tmp2b = geno2[split:length]
        geno1 = tmp1a + tmp2b
        geno2 = tmp2a + tmp1b
    return geno1, geno2
#----------------------------------------------------------------------------
def pikaia_mutate(npar, mutation_probability, geno, seeding):
    length = len(geno)
    tmp = ['']*length
    for i in range(length):
        tmp[i] = geno[i]
        numpy.random.seed(seeding)
        rnd = numpy.random.uniform(0, 1)
        
        if rnd < mutation_probability:
            numpy.random.seed(seeding)
            temp = numpy.random.uniform(0, 1)
            temp = str(int(temp*10.))
            tmp[i] = temp.replace(' ','')
    geno = ''
    for i in range(length):
        geno = geno + tmp[i]    
    return geno
#----------------------------------------------------------------------------
def pikaia_decode(npar, num_dec, geno):
    length = len(geno)
    tmp = ['']*length
    pheno = [0]*npar
    tmp_pheno = ['']*npar
    for i in range(length):
        tmp[i] = geno[i]    
    if length !=  npar*num_dec:  # assume that it is too short, pad with random digits
        adder = npar*num_dec - length
        for i in range(adder):
            temp = numpy.random.uniform(0, 1)
            temp = str(int(temp*10.))
            addstr = temp.replace(' ','')
            tmp = numpy.append(tmp, addstr)
    bot = 0
    for i in range(npar):
        top = bot + num_dec - 1
        tmp_pheno[i] = '0.'    
        for j in range(bot,top+1): 
            tmp_pheno[i] = tmp_pheno[i] + tmp[j]        
        pheno[i] = float(tmp_pheno[i])
        bot = bot + num_dec
    return pheno
#----------------------------------------------------------------------------
def pikaia_genrep(jj, pheno1, pheno2, new_phenotype, old_phenotype):
    i1 = 2*jj
    i2 = i1 + 1
    new_phenotype[i1,:] = pheno1 
    new_phenotype[i2,:] = pheno2
    return new_phenotype
#----------------------------------------------------------------------------
def pikaia_newpop(func, npop, oldphen, newphen, gid, fitness):
    # Test to see if we have to keep fittest of old generation
    tester = func(newphen[0])

    if tester < fitness[gid[0]]:
        newphen[0] = oldphen[gid[0]]
        
    oldphen = newphen  # Replace the population
        
    # Evaluate fitness of this new generation
    for i in range(npop):
        tmp = oldphen[i,:]
        fitness[i] = func(tmp)
    
    gid = numpy.argsort(fitness)[::-1]
    return oldphen, newphen, gid, fitness
#----------------------------------------------------------------------------
#
# Main Pikaia 
#
#----------------------------------------------------------------------------
def pikaia(mod, npar, ngen, npop, testinput):

    # load module and function optimizer.optimize to do satire calculation
    mod1 = __import__(mod[0])
    func = getattr(mod1, mod[1])
    
    # Pikaia parameters
    seeding = int(time.time())    # use current system time as seeding
    fitness_differential = 1.0    # Used in "roulette wheel" selection
    num_dec = 5                   # ndim - number of decimal places
    cross_probability = 0.85      # Probability that two "parents" are cross-bred
    mutation_probability = 0.01   # Probability that gene is mutated
    
    # Initialize the population at random
    numpy.random.seed(seeding)
    old_phenotype = numpy.random.uniform(0, 1, (npop, npar))
    old_phenotype[0,:] = testinput
    
    # Evaluate fitness of the initial population
    fit = [0]*npop
    for i in range(npop):
        tmp = old_phenotype[i,:]
        fit[i] = func(tmp)
    
    # Sort at this first step  - fittest(highest value) is the best, sorting in descending order
    gid = numpy.argsort(fit)[::-1]
    
    # create new phenotype array for manipulation/breeding
    new_phenotype = old_phenotype*0
    
    for ii in range(ngen):   # Being generation loop

        for jj in range(int(numpy.floor(npop/2))):   # Being population loop
                ########
                # 1. Pick two parents
                ########
                id1 = pikaia_select(npop, gid, fitness_differential)
                id2 = pikaia_select(npop, gid, fitness_differential)
                while id1 == id2:  # make sure two parents do not duplicate
                    id2 = pikaia_select(npop, gid, fitness_differential)
                ########
                # 2. Encode parents
                ########
                geno1 = pikaia_encode(npar, num_dec, old_phenotype[id1,:])
                geno2 = pikaia_encode(npar, num_dec, old_phenotype[id2,:])
                ########
                # 3. Breed  (Cross-Over & Mutate)
                ########
                geno1, geno2 = pikaia_crossover(npar, num_dec, cross_probability, geno1, geno2, seeding)
                geno1 = pikaia_mutate(npar, mutation_probability, geno1, seeding)
                geno2 = pikaia_mutate(npar, mutation_probability, geno2, seeding)
                ########
                # 4. Decode offspring genotypes
                ########
                pheno1 = pikaia_decode(npar, num_dec, geno1)
                pheno2 = pikaia_decode(npar, num_dec, geno2)
                ########
                # 5. Insert into population
                ########
                old_phenotype = copy.copy(old_phenotype)
                new_phenotype = pikaia_genrep(jj, pheno1, pheno2, new_phenotype, old_phenotype)
                ########
        # End population loop
        old_phenotype, new_phenotype, gid, fit = pikaia_newpop(func, npop, old_phenotype, new_phenotype, gid, fit)
        
        print('Generation : ', ii)
        print(f'Best   :  {gid[0]}  fitness:  {fit[gid[0]]:.5f}')
        print(f'{old_phenotype[gid[0],:]:.5f}')
        print('*****---------------------------------------------------*****')
    # End generation loop

















