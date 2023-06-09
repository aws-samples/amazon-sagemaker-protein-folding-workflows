#!/bin/bash
ldconfig

echo "$SM_HPS"

data_dir=$SM_CHANNEL_GENETIC
fasta_dir=$SM_CHANNEL_FASTA
fasta_path=${fasta_dir}/${SM_HP_FASTA_SUFFIX}
msa_path=$SM_CHANNEL_MSA

echo "data_dir is $data_dir"
echo "fasta_path is $fasta_path"
echo "msa_path is $msa_path"

ls $data_dir
ls $fasta_path
ls $msa_path

if [ $SM_HP_DB_PRESET == 'full_dbs' ]; then 
    args="--uniref30_database_path=${data_dir}/uniref30/UniRef30_2021_03 --bfd_database_path=${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
else
    args="--small_bfd_database_path=${data_dir}/small_bfd/bfd-first_non_consensus_sequences.fasta"
fi

WDIR=$SM_MODEL_DIR # output model directory, used as the working directory. Files in this dir will be package up in a model.tar.gz and upload to S3 at the end of the job

if [ ! -f ${msa_path}/model.tar.gz ]; then exit 1; fi

if [ -f ${msa_path}/model.tar.gz ]; then
    tar xfzv ${msa_path}/model.tar.gz -C $WDIR
fi

python /app/alphafold/run_alphafold.py \
--fasta_paths=${fasta_path} \
--uniref90_database_path=${data_dir}/uniref90/uniref90.fasta \
--mgnify_database_path=${data_dir}/mgnify/mgy_clusters_2022_05.fa \
--data_dir=$data_dir \
--template_mmcif_dir=${data_dir}/pdb_mmcif/mmcif_files \
--obsolete_pdbs_path=${data_dir}/pdb_mmcif/obsolete.dat \
--pdb70_database_path=${data_dir}/pdb70/pdb70 \
${args} \
--output_dir=$SM_MODEL_DIR \
--max_template_date=$SM_HP_MAX_TEMPLATE_DATE \
--db_preset=$SM_HP_DB_PRESET \
--model_preset=$SM_HP_MODEL_PRESET \
--benchmark=False \
--use_precomputed_msas=True \
--num_multimer_predictions_per_model=$SM_HP_NUM_MULTIMER_PREDICTIONS_PER_MODEL \
--use_gpu_relax=True \
--logtostderr