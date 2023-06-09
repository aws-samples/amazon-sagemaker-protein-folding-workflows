# Original Copyright 2021 DeepMind Technologies Limited
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
sys.path.append('/app/alphafold')
"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import shutil
import time
from typing import Dict, Union

from absl import app
from absl import flags
from absl import logging
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_integer('n_cpu', 8, 'The number of CPUs to give Jackhmmer and HHblits.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def generate_features(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
):
    """Predicts structure using AlphaFold for the given sequence."""
    logging.info('Processing %s', fasta_name)
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    # Get features.
    t_0 = time.time()

    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    timings['features'] = time.time() - t_0

    # Write out features as a pickled dictionary.
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)

    logging.info(
        f"Feature creation complete."
    )
    logging.info(f"Final timings for {fasta_name}: {timings}")
    timings_output_path = os.path.join(output_dir, "timings.json")
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))
    return

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
        if not FLAGS[f'{tool_name}_binary_path'].value:
            raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                            'sure it is installed on your system.')

    use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
    _check_flag('small_bfd_database_path', 'db_preset',
                should_be_set=use_small_bfd)
    _check_flag('bfd_database_path', 'db_preset',
                should_be_set=not use_small_bfd)
    _check_flag('uniref30_database_path', 'db_preset',
                should_be_set=not use_small_bfd)

    run_multimer_system = 'multimer' in FLAGS.model_preset
    _check_flag('pdb70_database_path', 'model_preset',
                should_be_set=not run_multimer_system)
    _check_flag('pdb_seqres_database_path', 'model_preset',
                should_be_set=run_multimer_system)
    _check_flag('uniprot_database_path', 'model_preset',
                should_be_set=run_multimer_system)

    # Check for duplicate FASTA file names.
    fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
    if len(fasta_names) != len(set(fasta_names)):
        raise ValueError('All FASTA paths must have a unique basename.')

    if run_multimer_system:
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=FLAGS.hmmsearch_binary_path,
            hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
            database_path=FLAGS.pdb_seqres_database_path)
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
    else:
        template_searcher = hhsearch.HHSearch(
            binary_path=FLAGS.hhsearch_binary_path,
            databases=[FLAGS.pdb70_database_path])
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd)

    if run_multimer_system:
        data_pipeline = pipeline_multimer.DataPipeline(
            monomer_data_pipeline=monomer_data_pipeline,
            jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
            uniprot_database_path=FLAGS.uniprot_database_path)
    else:
        data_pipeline = monomer_data_pipeline

    # Predict structure for each of the sequences.
    for i, fasta_path in enumerate(FLAGS.fasta_paths):
        fasta_name = fasta_names[i]

        generate_features(
            fasta_path=fasta_path,
            fasta_name=fasta_name,
            output_dir_base=FLAGS.output_dir,
            data_pipeline=data_pipeline,         
        )

if __name__ == '__main__':
    flags.mark_flags_as_required([
            'fasta_paths',
            'output_dir',
            'uniref90_database_path',
            'mgnify_database_path',
            'template_mmcif_dir',
            'max_template_date',
            'obsolete_pdbs_path',
        ])

    app.run(main)