import yaml


def dirname_from_params_path(param_file_path, struct):
    with open(param_file_path, 'r') as stream:
        d = yaml.safe_load(stream)
    return dirname_from_params(d, struct)

def dirname_from_params(d, struct):
    file_name_elt_lst = []
    if 'contact_sequence_file' in d:
        traj_name = d['contact_sequence_file'].split('/')[-1].split('.cs')[0]
        file_name_elt_lst.append(traj_name)
    else:
        traj_name = d['data_file_path'].split('/')[-1].split('.npz')[0]
        file_name_elt_lst.append(traj_name)
    if 'noisy_measurements' in d:
        noise = 'noise' if d['noisy_measurements'] else 'nonoise'
        file_name_elt_lst.append(traj_name)
    if 'scale_dist' in d:
        scale = str(d['scale_dist'])
        mass_dist = 'mass' if d['mass_dist'] else 'lever'
        base_dist = 'basedist' if d['base_dist_only'] else 'alldist'
        file_name_elt_lst.append(base_dist)
        disturbed_inertial_meas = '~'
        if d['vbc_is_dist']: disturbed_inertial_meas += 'V'
        if d['Iw_is_dist']: disturbed_inertial_meas += 'I'
        if d['Lgest_is_dist']: disturbed_inertial_meas += 'L'
        file_name_elt_lst.append(scale)
        file_name_elt_lst.append(mass_dist)
        file_name_elt_lst.append(disturbed_inertial_meas)

    bdrift = 'drift' if d['std_bp_drift'] > 1 else 'nodrift'
    file_name_elt_lst.append(bdrift)

    file_name = '_'.join(file_name_elt_lst)

    return file_name