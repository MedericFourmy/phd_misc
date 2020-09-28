import yaml


def dirname_from_params_path(param_file_path):
    with open(param_file_path, 'r') as stream:
        d = yaml.safe_load(stream)
    return dirname_from_params(d)

def dirname_from_params(d):
    struct = d['structure']
    traj_name = d['contact_sequence_file'].split('/')[-1].split('.cs')[0]
    noise = 'noise' if d['noisy_measurements'] else 'nonoise'
    scale = str(d['scale_dist'])
    bdrift = 'drift' if d['std_bp_drift'] > 1 else 'nodrift'
    base_dist = 'basedist' if d['base_dist_only'] else 'alldist'
    mass_dist = 'mass' if d['mass_dist'] else 'lever'
    disturbed_inertial_meas = '~'
    if d['vbc_is_dist']: disturbed_inertial_meas += 'V'
    if d['Iw_is_dist']: disturbed_inertial_meas += 'I'
    if d['Lgest_is_dist']: disturbed_inertial_meas += 'L'

    file_name = '_'.join([struct, traj_name, noise, bdrift, base_dist, mass_dist, scale, disturbed_inertial_meas])
    return file_name