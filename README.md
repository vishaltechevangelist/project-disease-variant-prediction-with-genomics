1. conda create -n genomics python=3.10 -y
2. conda activate genomics
3. conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn
4. conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
5. conda install -c bioconda cyvcf2 pyfaidx biopython
6. pip install transformers datasets sentencepiece


bcftools view clinvar_20250923.vcf -Ov -o cleaned.vcf
