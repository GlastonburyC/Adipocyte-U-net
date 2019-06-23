#!/bin/sh
#$ -cwd

## data is already imputed
## files in the format chr*.imputed.poly.subset300.[bim/bed/fam]

data=/data/EXTEND_frayling_adipocyte_genotypes
home=/home/sara/histology/frayling
plink=/home/plink/1.90b3/plink

## Generate a bunch of stats on the samples and snps
for chr in {1..23}; do

    ## sample missingness
    $plink --bfile $data/chr$chr.imputed.poly.subset300 --allow-no-sex \
	    --missing \
	    --threads 1 \
	    --out $home/frayling.chr$chr.imputed.poly.subset300

    ## SNPs for snp super set
    $plink --bfile $data/chr$chr.imputed.poly.subset300 --allow-no-sex \
	    --geno 0.001 \
	    --hwe 1e-3 \
	    --maf 0.01 \
	    --indep-pairwise 50 5 0.2 \
	    --threads 1 \
	    --out $home/frayling.chr$chr.imputed.poly.subset300

done

## SNPs to drop from super set -- lactase gene
awk ' $1==2 && $4 > 129883530 && $4 < 140283530 { print $2 } ' $data/chr2.imputed.poly.subset300.bim > ./frayling.imputed.poly.subset300.pca.EXCLUDE.snps

## Drop all of the SNPs in the major histocompatibility complex (hg19 coordinates)
awk ' $1==6 && $4 > 24092021 && $4 < 38892022 { print $2 } ' $data/chr6.imputed.poly.subset300.bim >> ./frayling.imputed.poly.subset300.pca.EXCLUDE.snps

## Drop all of the SNPs in the inverted regions on chromosomes 8 and 17
awk ' $1==8 && $4 > 6612592 && $4 < 13455629 { print $2 } ' $data/chr8.imputed.poly.subset300.bim >> ./frayling.imputed.poly.subset300.pca.EXCLUDE.snps
awk ' $1==17 && $4 > 40546474 && $4 < 44644684 { print $2 } ' $data/chr17.imputed.poly.subset300.bim >> ./frayling.imputed.poly.subset300.pca.EXCLUDE.snps

## Extract the super set while removing the above SNPs; autosome only
for chr in {1..22}; do

    $plink --bfile $data/chr$chr.imputed.poly.subset300 --allow-no-sex \
	    --threads 1 \
	    --maf 0.10 \
	    --extract $home/frayling.chr$chr.imputed.poly.subset300.prune.in \
	    --exclude ./frayling.imputed.poly.subset300.pca.EXCLUDE.snps \
	    --make-bed \
	    --out $home/chr$chr.imputed.poly.subset300.pca

done

## merge all the files together
$plink --bfile chr1.imputed.poly.subset300.pca --merge-list frayling.imputed.poly.subset300.pca.merge.lst \
    --make-bed --out merged.imputed.poly.subset300.pca --allow-no-sex

## remove inconsistent alleles (and repeat merge)
for chr in {1..22}; do
  $plink --bfile chr$chr.imputed.poly.subset300.pca --exclude merged.imputed.poly.subset300.pca-merge.missnp --allow-no-sex --make-bed --out chr$chr.imputed.poly.subset300.pca
done

## inbreeding
$plink --bfile merged.imputed.poly.subset300.pca --allow-no-sex --het --out merged.imputed.poly.subset300.pca

## relatedness
$plink --bfile merged.imputed.poly.subset300.pca --allow-no-sex --genome --out merged.imputed.poly.subset300.pca

## run sex check
$plink --bfile $data/chr23.imputed.poly.subset300 --update-sex fATDIVA_BMI_Data_Pairs.08.11.17.sex.txt \
    --make-bed --out $home/chr23.imputed.poly.subset300
$plink --bfile $home/chr23.imputed.poly.subset300 --check-sex --allow-no-sex --out $home/chr23.imputed.poly.subset300
