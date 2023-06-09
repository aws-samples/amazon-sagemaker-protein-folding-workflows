import argparse
import os, sys
import json
import tarfile
import warnings
import pymol
from pymol import cmd, stored
import numpy as np
# from sagemaker.experiments.run import load_run


def rmsd_cur(mol0, mol1, sel='*'):
    """
    Computes the root mean square deviation from the current
    coordinates of two pairs of equivalent atoms. Does not
    perform a superposition.
    
    Parameters
    ----------
    mol0 : PyMOL object
    mol1 : PyMOL object
    sel  : PyMOL selection, atoms used to compute rmsd.
           e.g. use ca+c+n for the backbone
    """
    model0 = cmd.get_model('%s and name %s' % (mol0, sel))
    model1 = cmd.get_model('%s and name %s'  % (mol1, sel))
    xyz0 = np.array(model0.get_coord_list())
    xyz1 = np.array(model1.get_coord_list())
    
    rmsd = (np.sum((xyz0 - xyz1 )**2)/len(xyz0))**0.5
    return rmsd


def rmsd_fit(mol0, mol1, sel='*', fit=True):
    """
    Computes the root mean square deviation from two pairs of
    equivalent atoms after superposition.
    
    Parameters
    ----------
    mol0 : PyMOL object
    mol1 : PyMOL object
    sel  : PyMOL selection. atoms used to compute rmsd.
           e.g. use ca+c+n for the backbone
    fit  : bool. If false computes the rmsd after superposition, but without
           updating the coordinates
           
    """
    xyz0 = np.array(cmd.get_model('%s and name %s' % (mol0, sel)).get_coord_list())
    xyz1 = np.array(cmd.get_model('%s and name %s'  % (mol1, sel)).get_coord_list())
    
    xyz0_all = np.array(cmd.get_model('%s' % mol0).get_coord_list())
    xyz1_all = np.array(cmd.get_model('%s'  % mol1).get_coord_list())
    
    # Translation
    X = xyz0 - xyz0.mean(axis=0)
    Y = xyz1 - xyz1.mean(axis=0)
    # Covariation matrix
    Cov_matrix = np.dot(Y.T, X)
    # Optimal rotation matrix
    U, S, Wt = np.linalg.svd(Cov_matrix)
    # Create Rotation matrix R
    R = np.dot(U, Wt)
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0.:
        S[-1] = -S[-1]
        Wt[-1] *= -1
        R = np.dot(U, Wt) 
    if fit:
        # center the first molecule
        stored.sel0 = list(xyz0_all - xyz0.mean(axis=0))
        # rotate and translate the second molecule
        stored.sel1 = list(np.dot((xyz1_all - xyz1.mean(0)), R))
        #update the changes to the coordinates 
        cmd.alter_state(1, mol0,"(x,y,z)=stored.sel0.pop(0)")
        cmd.alter_state(1, mol1,"(x,y,z)=stored.sel1.pop(0)")

    # We compute the RMSD after superposition by using the matrix S. The advantage is 
    # we do not need to actually do the superposition before computing the RMSD.
    #rmsd = (np.exp(np.log(np.sum(X ** 2) + np.sum(Y ** 2)) - 2.0 * np.log(np.sum(S)))/len(X))**0.5
    rmsd = ((np.sum(X ** 2) + np.sum(Y ** 2) - 2.0 * np.sum(S))/len(X))**0.5
    # scales and translates the window to show a selection
    cmd.zoom()
    return rmsd


def tm_score(mol0, mol1, sel='*'): #Check if TM-align use all atoms!
    """
    Compute TM-score between two set of coordinates
    
    Parameters
    ----------
    mol0 : PyMOL object
    mol1 : PyMOL object
    sel  : PyMOL selection, atoms used to compute rmsd.
           e.g. use ca+c+n for the backbone
    """
    xyz0 = np.array(cmd.get_model('%s and name %s' % (mol0, sel)).get_coord_list())
    xyz1 = np.array(cmd.get_model('%s and name %s'  % (mol1, sel)).get_coord_list())
    
    L = len(xyz0)
    # d0 is less than 0.5 for L < 22 
    # and nan for L < 15 (root of a negative number)
    d0 = 1.24 * np.power(L - 15, 1/3) - 1.8
    d0 = max(0.5, d0) 

    # compute the distance for each pair of atoms
    di = np.sum((xyz0 - xyz1) ** 2, 1) # sum along first axis
    return np.sum(1 / (1 + (di / d0) ** 2)) / L


# what to be the input?
# two model.tar.gz S3 URI? Then they will be available in the instance /opt/ml/processing/{method1,method2}
# but how do you access the pdb file? when each method has its own folder structure and naming
# perhaps another argument/ENV to specify the pdb file to analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb1', type=str, help='PDB file name for the first structure to compare that can be found in the corresponding tar.gz file, e.g. ranked_0.pdb.')
    parser.add_argument('--pdb2', type=str, help='PDB file name for the second structure to compare that can be found in the corresponding tar.gz file, e.g. XXX_model_1_ptm_relaxed.pdb.')
    args, _ = parser.parse_known_args()
    
    print("Received arguments {}".format(args))
    
    processing_dir='/opt/ml/processing'
    # processing_dir='/root/protein-folding-on-sagemaker'

    # open file
    tarfile1 = os.path.join(processing_dir, "input-pdb1", "model.tar.gz")
    file1 = tarfile.open(tarfile1)
    # print file names
    print(file1.getnames())
    foi1=[i for i in file1.getnames() if i.endswith(args.pdb1)]
    # extract files
    file1.extract(foi1[0], path=os.path.join(processing_dir, "input-pdb1"))
    # close file
    file1.close()
    
    # for the second file
    tarfile2 = os.path.join(processing_dir, "input-pdb2", "model.tar.gz")
    file2 = tarfile.open(tarfile2)
    print(file2.getnames())
    foi2=[i for i in file2.getnames() if i.endswith(args.pdb2)]
    # extract files
    file2.extract(foi2[0], path=os.path.join(processing_dir, "input-pdb2"))
    file2.close()
    
    pdbfile1=os.path.join(processing_dir, "input-pdb1", foi1[0])
    pdbfile2=os.path.join(processing_dir, "input-pdb2", foi2[0])
    
    cmd.load(pdbfile1)
    cmd.load(pdbfile2)
    cmd.remove('not polymer or hydro')
    object1 = cmd.get_names()[0]
    object2 = cmd.get_names()[1]
    print(object1)
    print(object2)
    
    # compute rmsd_cur
    rmsd = rmsd_cur(object1, object2, sel='ca') #'ca+c+n'
    print('%.2f' % rmsd)
    
    # compute rmsd_fit
    rmsd_sp = rmsd_fit(object1, object2, sel='ca', fit=True)
    print('%.2f' % rmsd_sp)
    
    # compute TM score
    tmscore = tm_score(object1, object2, sel='ca')
    print('%.4f' % tmscore)
    
    report_dict = {
        "structure_metrics": {
            "rmsd": {"value": rmsd},
            "rmsd_superposition": {"value": rmsd_sp},
            "tm_score": {"value": tmscore}
        },
    }
    
    output_dir = os.path.join(processing_dir, 'evaluation')
    # output_dir = '/root/protein-folding-on-sagemaker/evaluation'
    os.makedirs(output_dir, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        