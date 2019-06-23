#!/bin/sh
#$ -cwd
#$ -S /bin/bash
#$ -l h_rt=24:00:00,h_vmem=10G
#$ -pe threaded 1

#Pointers to programs
CONVERTF_EXEC=/apps/well/eigensoft/5.0.2/bin/convertf
SMARTPCA_EXEC=/apps/well/eigensoft/5.0.2/bin/smartpca
EIGENSTRAT_EXEC=/apps/well/eigensoft/5.0.2/bin/eigenstrat
PED_ROOT=$1

# MUST put smartpca bin directory in path for smartpca.perl to work
PATH="/hpc/local/CentOS6/cog_gonl/software/eig4.2m/EIG4.2/bin:$PATH"
##Convert to eigenstrat format using convertf##

echo "genotypename:    $PED_ROOT.ped" > $PED_ROOT.convertf.par
echo "snpname:         $PED_ROOT.map" >> $PED_ROOT.convertf.par
##echo "indivname:       $PED_ROOT.pedind" >> $PED_ROOT.convertf.par ## Comment this in and comment out the next line for PCA projection
echo "indivname:       $PED_ROOT.ped" >> $PED_ROOT.convertf.par
echo "outputformat:    EIGENSTRAT" >> $PED_ROOT.convertf.par
echo "genotypeoutname: $PED_ROOT.geno" >> $PED_ROOT.convertf.par
echo "snpoutname:      $PED_ROOT.snp" >> $PED_ROOT.convertf.par
echo "indivoutname:    $PED_ROOT.ind" >> $PED_ROOT.convertf.par
echo "familynames:     NO" >> $PED_ROOT.convertf.par

$CONVERTF_EXEC -p $PED_ROOT.convertf.par

### Calculate eigenvectors

##echo "ref"             > $PED_ROOT.POPlist            ## Comment this in for PCA projection
echo "genotypename:    $PED_ROOT.geno" > $PED_ROOT.smartpca.par
echo "snpname:         $PED_ROOT.snp" >> $PED_ROOT.smartpca.par
##echo "indivname:       $PED_ROOT.pedind" >> $PED_ROOT.smartpca.par ## Comment this in and comment out the next line for PCA projection
echo "indivname:       $PED_ROOT.ped" >> $PED_ROOT.smartpca.par
echo "evecoutname:     $PED_ROOT.evec" >> $PED_ROOT.smartpca.par
echo "evaloutname:     $PED_ROOT.eval" >> $PED_ROOT.smartpca.par
echo "altnormstyle:    NO" >> $PED_ROOT.smartpca.par
echo "numoutevec:      100" >> $PED_ROOT.smartpca.par
echo "numoutlieriter:  0"  >> $PED_ROOT.smartpca.par
##echo "poplistname:     $PED_ROOT.POPlist" >> $PED_ROOT.smartpca.par ## Comment this in for PCA projection

$SMARTPCA_EXEC -p $PED_ROOT.smartpca.par > $PED_ROOT.smartpca.log
