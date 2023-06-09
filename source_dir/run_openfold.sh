#!/bin/bash -vx

echo "$SM_HPS"

data_dir=$SM_CHANNEL_GENETIC
fasta_path=$SM_CHANNEL_FASTA
msa_path=$SM_CHANNEL_MSA
openfold_checkpoint_path=$SM_CHANNEL_PARAM

echo "data_dir is $data_dir"
echo "fasta_path is $fasta_path"
echo "msa_path is $msa_path"
echo "openfold_checkpoint_path is $openfold_checkpoint_path"

ls $data_dir
ls $fasta_path
ls $openfold_checkpoint_path

WDIR=$SM_MODEL_DIR # output model directory, used as the working directory. Files in this dir will be package up in a model.tar.gz and upload to S3 at the end of the job

if [ ! -f ${msa_path}/model.tar.gz ]; then exit 1; fi

if [ -f ${msa_path}/model.tar.gz ]; then
    tar xfzv ${msa_path}/model.tar.gz -C $WDIR
    cd $WDIR/msas/
    mv * ../
fi

python /opt/openfold/run_pretrained_openfold.py \
${fasta_path} \
${data_dir}/pdb_mmcif/mmcif_files/ \
--model_device "cuda:0" \
--jackhmmer_binary_path /opt/conda/bin/jackhmmer \
--hhblits_binary_path /opt/conda/bin/hhblits \
--hhsearch_binary_path /opt/conda/bin/hhsearch \
--kalign_binary_path /opt/conda/bin/kalign \
--config_preset "model_1_ptm" \
--openfold_checkpoint_path ${openfold_checkpoint_path}/finetuning_ptm_2.pt \
--preset $SM_HP_DB_PRESET \
--output_dir $SM_MODEL_DIR \
--use_precomputed_alignments $SM_MODEL_DIR/