#!/bin/bash
ldconfig

echo "$SM_HPS"

data_dir=$SM_CHANNEL_GENETIC
fasta_dir=$SM_CHANNEL_FASTA
fasta_path=${fasta_dir}/${SM_HP_FASTA_SUFFIX}

echo "data_dir is $data_dir"
echo "fasta_path is $fasta_path"

ls $data_dir
ls $fasta_path
    
if [ $SM_HP_DB_PRESET == 'full_dbs' ]; then 
    args="--uniref30_database_path=${data_dir}/uniref30/UniRef30_2021_03 --bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
else
    args="--small_bfd_database_path=${data_dir}/small_bfd/bfd-first_non_consensus_sequences.fasta"
fi
        
python create_alignments.py \
--fasta_paths=${fasta_path} \
--uniref90_database_path=${data_dir}/uniref90/uniref90.fasta \
--mgnify_database_path=${data_dir}/mgnify/mgy_clusters_2022_05.fa \
--template_mmcif_dir=${data_dir}/pdb_mmcif/mmcif_files \
--obsolete_pdbs_path=${data_dir}/pdb_mmcif/obsolete.dat \
--pdb70_database_path=${data_dir}/pdb70/pdb70 \
${args} \
--output_dir=$SM_MODEL_DIR \
--max_template_date=$SM_HP_MAX_TEMPLATE_DATE \
--db_preset=$SM_HP_DB_PRESET \
--model_preset=$SM_HP_MODEL_PRESET \
--logtostderr 