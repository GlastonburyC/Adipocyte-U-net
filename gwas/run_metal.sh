#!/bin/sh
#$ -cwd
#$ -l h_rt=01:00:00

depot=$1
sex=$2
pheno=$3

metal=/home/metal/20110325/metal

echo "=== Beginning meta-analysis ==="
echo "=== Running: $depot $sex $pheno ==="

$metal ./$depot.$sex.$pheno.metalParams.txt

mv ./METAANALYSIS1.TBL ./$depot/$sex/histology.meta-analysis.$depot.$sex.$pheno.tbl
mv ./METAANALYSIS1.TBL.info ./$depot/$sex/histology.meta-analysis.$depot.$sex.$pheno.tbl.info

echo "=== $depot $sex $pheno analysis complete ==="
