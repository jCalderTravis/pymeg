'''
Compute source reconstruction for all subjects

Succesfull source reconstruction depends on a few things:

fMRI
    1. Recon-all MRI for each subject
    2. mne watershed_bem for each subject
    3. mne make_scalp_surfaces for all subjects
    4. Coreg the Wang & Kastner atlas to each subject using the scripts
       in require/apply_occ_wang/bin (apply_template and to_label)
MEG:
    5. A piece of sample data for each subject (in fif format)
    6. Create a coregistration for each subjects MRI with mne coreg
    7. Create a source space
    8. Create a bem model
    9. Create a leadfield
   10. Compute a noise and data cross spectral density matrix
       -> Use a data CSD that spans the entire signal duration.
   11. Run DICS to project epochs into source space
   12. Extract time course in labels from Atlas.

'''
import os
from os.path import join
import mne
import pandas as pd
import numpy as np

from joblib import Memory
from conf_analysis.behavior import metadata

memory = Memory(cachedir=metadata.cachedir, verbose=0)
subjects_dir = os.environ['SUBJECTS_DIR']


def set_fs_subjects_dir(directory):
    global subjects_dir
    os.environ['SUBJECTS_DIR'] = directory
    subjects_dir = directory


#trans_dir = '/home/nwilming/conf_meg/trans'
#plot_dir = '/home/nwilming/conf_analysis/plots/source'


def check_bems(subjects):
    '''
    Create a plot of all BEM segmentations
    '''
    for sub in subjects:
        fig = mne.viz.plot_bem(subject=sub,
                               subjects_dir=subjects_dir,
                               brain_surfaces='white',
                               orientation='coronal')


@memory.cache
def get_source_space(subject):
    '''
    Return source space.
    Abstract this here to have it unique for everyone.
    '''
    return mne.setup_source_space(subject, spacing='oct6',
                                  subjects_dir=subjects_dir,
                                  add_dist=False)


def get_trans(subject, session):
    '''
    Return filename of transformation for a subject
    '''
    file_ident = 'S%i-SESS%i' % (subject, session)
    return join(trans_dir, file_ident + '-trans.fif')


@memory.cache
def get_info(filename):
    '''
    Return an info dict for a measurement from one subject/session.

    Parameters
        filename : string
    Path of a data set for subject/session.
    '''
    trans, fiducials, info = get_head_correct_info(
        subject, session)
    return info


@memory.cache
def get_leadfield(subject, filename, conductivity):
    '''
    Compute leadfield with presets for this subject

    Parameters
        subject : string
        filename : string

    '''
    src = get_source_space(subject)
    model = make_fieldtrip_bem_model(
        subject=subject,
        ico=None,
        conductivity=conductivity,
        subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    trans = get_trans(filename)
    info = get_info(filename)

    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=2)
    return fwd, bem, fwd['src'], trans


def make_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3),
                             subjects_dir=None, verbose=None, bem_sub_path='bem_ft'):
    """Create a BEM model for a subject.

    Copied from MNE python, adapted to read surface from fieldtrip / spm
    segmentation.
    """
    import os.path as op
    from mne.io.constants import FIFF
    conductivity = np.array(conductivity, float)
    if conductivity.ndim != 1 or conductivity.size not in (1, 3):
        raise ValueError('conductivity must be 1D array-like with 1 or 3 '
                         'elements')
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    subject_dir = op.join(subjects_dir, subject)
    bem_dir = op.join(subject_dir, bem_sub_path)
    inner_skull = op.join(bem_dir, 'inner_skull.surf')
    outer_skull = op.join(bem_dir, 'outer_skull.surf')
    outer_skin = op.join(bem_dir, 'outer_skin.surf')
    surfaces = [inner_skull, outer_skull, outer_skin]
    ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN,
           FIFF.FIFFV_BEM_SURF_ID_SKULL,
           FIFF.FIFFV_BEM_SURF_ID_HEAD]
    print('Creating the BEM geometry...')
    if len(conductivity) == 1:
        surfaces = surfaces[:1]
        ids = ids[:1]
    surfaces = mne.bem._surfaces_to_bem(surfaces, ids, conductivity, ico)
    mne.bem._check_bem_size(surfaces)
    return surfaces


@memory.cache
def get_labels(subject, filters=['*wang2015atlas*', '*a2009s*.label']):
    import glob
    subject_dir = join(subjects_dir, subject)
    labels = []
    for filter in filters:
        labels += glob.glob(join(subject_dir, 'label', filter))
    return [mne.read_label(label, subject) for label in labels]


def add_volume_info(subject, surface, subjects_dir, volume='T1'):
    """Add volume info from MGZ volume
    """
    import os.path as op
    from mne.bem import _extract_volume_info
    from mne.surface import (read_surface, write_surface)
    subject_dir = op.join(subjects_dir, subject)
    mri_dir = op.join(subject_dir, 'mri')
    T1_mgz = op.join(mri_dir, volume + '.mgz')
    new_info = _extract_volume_info(T1_mgz)
    print new_info.keys()
    rr, tris, volume_info = read_surface(surface,
                                         read_metadata=True)

    # volume_info.update(new_info)  # replace volume info, 'head' stays
    print volume_info.keys()
    import numpy as np
    if 'head' not in volume_info.keys():
        volume_info['head'] = np.array([2,  0, 20], dtype=np.int32)
    write_surface(surface, rr, tris, volume_info=volume_info)


'''
Transformation matrix MEG<>T1 space.
'''


@memory.cache
def get_head_correct_info(filename, N=-1):
    trans = get_ctf_trans(filename)
    fiducials = get_ref_head_pos(filename, trans, N=N)
    raw = mne.io.ctf.read_raw_ctf(filename)
    info = replace_fiducials(raw.info, fiducials)
    return trans, fiducials, info


def make_trans(subject, filename, trans_name):
    '''
    Create coregistration between MRI and MEG space.
    '''
    import os
    import time
    import tempfile
    trans, fiducials, info = get_head_correct_info(filename)

    with tempfile.NamedTemporaryFile() as hs_ref:
        # hs_ref = '/home/nwilming/conf_meg/trans/S%i-SESS%i.fif' % (
        #    subject, session)
        mne.io.meas_info.write_info(hs_ref, info)

        if os.path.isfile(trans_name):
            raise RuntimeError(
                'Transformation matrix %s already exists' % trans_name)

        print('--------------------------------')
        print('Please save trans file as:')
        print(trans_name)

        cmd = 'mne coreg --high-res-head -d %s -s %s -f %s' % (
            '/home/nwilming/fs_subject_dir', 'S%02i' % subject, hs_ref)
        print cmd
        os.system(cmd)
        mne.gui.coregistration(inst=hs_ref, subject,
                               subjects_dir=subjects_dir)
        while not os.path.isfile(trans_name):
            print('Waiting for transformation matrix to appear')
            time.sleep(1)


@memory.cache
def get_ref_head_pos(filename,  trans, N=-1):
    from mne.transforms import apply_trans
    data = pymegprepr.load_epochs([data])[0]
    cc = head_loc(data.decimate(10))
    nasion = np.stack([c[0] for c in cc[:N]]).mean(0)
    lpa = np.stack([c[1] for c in cc[:N]]).mean(0)
    rpa = np.stack([c[2] for c in cc[:N]]).mean(0)
    nasion, lpa, rpa = nasion.mean(-1), lpa.mean(-1), rpa.mean(-1)

    return {'nasion': apply_trans(trans['t_ctf_dev_dev'], np.array(nasion)),
            'lpa': apply_trans(trans['t_ctf_dev_dev'], np.array(lpa)),
            'rpa': apply_trans(trans['t_ctf_dev_dev'], np.array(rpa))}


def replace_fiducials(info, fiducials):
    from mne.io import meas_info
    fids = meas_info._make_dig_points(**fiducials)
    info = info.copy()
    dig = info['dig']
    for i, d in enumerate(dig):
        if d['kind'] == 3:
            if d['ident'] == 3:

                dig[i]['r'] = fids[2]['r']
            elif d['ident'] == 2:
                dig[i]['r'] = fids[1]['r']
            elif d['ident'] == 1:
                dig[i]['r'] = fids[0]['r']
    info['dig'] = dig
    return info


def head_movement(epochs):
    ch_names = np.array(epochs.ch_names)
    channels = {'x': ['HLC0011', 'HLC0012', 'HLC0013'],
                'y': ['HLC0021', 'HLC0022', 'HLC0023'],
                'z': ['HLC0031', 'HLC0032', 'HLC0033']}
    channel_ids = {}
    for key, names in channels.iteritems():
        ids = [np.where([n in ch for ch in ch_names])[0][0] for n in names]
        channel_ids[key] = ids

    data = epochs._data
    ccs = []
    for e in range(epochs._data.shape[0]):
        x = np.stack([data[e, i, :] for i in channel_ids['x']])
        y = np.stack([data[e, i, :] for i in channel_ids['y']])
        z = np.stack([data[e, i, :] for i in channel_ids['z']])
        cc = circumcenter(x, y, z)
        ccs.append(cc)
    return np.stack(ccs)


@memory.cache
def get_head_loc(epochs):
    cc = head_loc(epochs)
    trans, fiducials, info = get_head_correct_info(subject, session)
    nose_coil = np.concatenate([c[0] for c in cc], -1)
    left_coil = np.concatenate([c[1] for c in cc], -1)
    right_coil = np.concatenate([c[2] for c in cc], -1)
    nose_coil = apply_trans(trans['t_ctf_dev_dev'], nose_coil.T)
    left_coil = apply_trans(trans['t_ctf_dev_dev'], left_coil.T)
    right_coil = apply_trans(trans['t_ctf_dev_dev'], right_coil.T)

    nose_coil = (nose_coil**2).sum(1)**.5
    left_coil = (left_coil**2).sum(1)**.5
    right_coil = (right_coil**2).sum(1)**.5
    return nose_coil, left_coil, right_coil


def head_loc(epochs):
    ch_names = np.array(epochs.ch_names)
    channels = {'x': ['HLC0011', 'HLC0012', 'HLC0013'],
                'y': ['HLC0021', 'HLC0022', 'HLC0023'],
                'z': ['HLC0031', 'HLC0032', 'HLC0033']}
    channel_ids = {}
    for key, names in channels.iteritems():
        ids = [np.where([n in ch for ch in ch_names])[0][0] for n in names]
        channel_ids[key] = ids

    data = epochs._data
    ccs = []
    if len(epochs._data.shape) > 2:
        for e in range(epochs._data.shape[0]):
            x = np.stack([data[e, i, :] for i in channel_ids['x']])
            y = np.stack([data[e, i, :] for i in channel_ids['y']])
            z = np.stack([data[e, i, :] for i in channel_ids['z']])
            ccs.append((x, y, z))
    else:
        x = np.stack([data[i, :] for i in channel_ids['x']])
        y = np.stack([data[i, :] for i in channel_ids['y']])
        z = np.stack([data[i, :] for i in channel_ids['z']])
        ccs.append((x, y, z))
    return ccs


def get_ctf_trans(filename):
    from mne.io.ctf.res4 import _read_res4
    from mne.io.ctf.hc import _read_hc
    from mne.io.ctf.trans import _make_ctf_coord_trans_set

    res4 = _read_res4(filename)  # Read the magical res4 file
    coils = _read_hc(filename)  # Read the coil locations

    # Investigate the coil location data to get the coordinate trans
    coord_trans = _make_ctf_coord_trans_set(res4, coils)
    return coord_trans


def circumcenter(coil1, coil2, coil3):
    # Adapted from:
    #    http://www.fieldtriptoolbox.org/example/how_to_incorporate_head_movements_in_meg_analysis
    # CIRCUMCENTER determines the position and orientation of the circumcenter
    # of the three fiducial markers (MEG headposition coils).
    #
    # Input: X,y,z-coordinates of the 3 coils [3 X N],[3 X N],[3 X N] where N
    # is timesamples/trials.
    #
    # Output: X,y,z-coordinates of the circumcenter [1-3 X N], and the
    # orientations to the x,y,z-axes [4-6 X N].
    #
    # A. Stolk, 2012

    # number of timesamples/trials
    N = coil1.shape[1]
    cc = np.zeros((6, N)) * np.nan
    # x-, y-, and z-coordinates of the circumcenter
    # use coordinates relative to point `a' of the triangle
    xba = coil2[0, :] - coil1[0, :]
    yba = coil2[1, :] - coil1[1, :]
    zba = coil2[2, :] - coil1[2, :]
    xca = coil3[0, :] - coil1[0, :]
    yca = coil3[1, :] - coil1[1, :]
    zca = coil3[2, :] - coil1[2, :]

    # squares of lengths of the edges incident to `a'
    balength = xba * xba + yba * yba + zba * zba
    calength = xca * xca + yca * yca + zca * zca

    # cross product of these edges
    xcrossbc = yba * zca - yca * zba
    ycrossbc = zba * xca - zca * xba
    zcrossbc = xba * yca - xca * yba

    # calculate the denominator of the formulae
    denominator = 0.5 / (xcrossbc * xcrossbc + ycrossbc * ycrossbc
                         + zcrossbc * zcrossbc)

    # calculate offset (from `a') of circumcenter
    xcirca = ((balength * yca - calength * yba) * zcrossbc -
              (balength * zca - calength * zba) * ycrossbc) * denominator
    ycirca = ((balength * zca - calength * zba) * xcrossbc -
              (balength * xca - calength * xba) * zcrossbc) * denominator
    zcirca = ((balength * xca - calength * xba) * ycrossbc -
              (balength * yca - calength * yba) * xcrossbc) * denominator

    cc[0, :] = xcirca + coil1[0, :]
    cc[1, :] = ycirca + coil1[1, :]
    cc[2, :] = zcirca + coil1[2, :]
    # orientation of the circumcenter with respect to the x-, y-, and z-axis
    # coordinates
    v = np.stack([cc[0, :].T, cc[1, :].T, cc[2, :].T]).T
    vx = np.stack([np.zeros((N,)).T, cc[1, :].T, cc[2, :].T]).T
    # on the x - axis
    vy = np.stack([cc[0, :].T, np.zeros((N,)).T, cc[2, :].T]).T
    # on the y - axis
    vz = np.stack([cc[0, :].T, cc[1, :].T, np.zeros((N,)).T]).T
    # on the z - axis
    thetax, thetay = np.zeros((N,)) * np.nan, np.zeros((N,)) * np.nan
    thetaz = np.zeros((N,)) * np.nan
    for j in range(N):

        # find the angles of two vectors opposing the axes
        thetax[j] = np.arccos(np.dot(v[j, :], vx[j, :]) /
                              (np.linalg.norm(v[j, :]) * np.linalg.norm(vx[j, :])))
        thetay[j] = np.arccos(np.dot(v[j, :], vy[j, :]) /
                              (np.linalg.norm(v[j, :]) * np.linalg.norm(vy[j, :])))
        thetaz[j] = np.arccos(np.dot(v[j, :], vz[j, :]) /
                              (np.linalg.norm(v[j, :]) * np.linalg.norm(vz[j, :])))

        # convert to degrees
        cc[3, j] = (thetax[j] * (180 / np.pi))
        cc[4, j] = (thetay[j] * (180 / np.pi))
        cc[5, j] = (thetaz[j] * (180 / np.pi))
    return cc


def ensure_iter(input):
    if isinstance(input, basestring):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input


def clear_cache():
    memory.clear()