import argparse
import logging
import os
from random import uniform
from time import sleep
from pathlib import Path
from typing import Optional

logging.basicConfig(level = logging.INFO, format = '[%(asctime)s][%(levelname)s] %(message)s', datefmt = r'%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('CoCo')

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'CoCo: Export CryoPROS-generated particles to CryoSPARC and initiate 2D classification')
    parser.add_argument(
        '-i', '--i', '--input',
        type = Path,
        required = True,
        help = 'Input STAR file (particle meta path)'
    )
    parser.add_argument(
        '-d', '--d', '--directory',
        type = Path,
        help = 'Directory containing particle data (particle data path)'
    )
    parser.add_argument(
        '--user',
        type = str,
        required = True,
        help = 'Email address of the CryoSPARC user'
    )
    parser.add_argument(
        '--project',
        type = str,
        required = True,
        help = 'Project UID in CryoSPARC'
    )
    parser.add_argument(
        '--workspace',
        type = str,
        required = True,
        help = 'Workspace UID in CryoSPARC'
    )
    parser.add_argument(
        '--lane',
        type = str,
        required = True,
        help = 'Compute lane to use'
    )
    return vars(parser.parse_args())

def load_cryosparc_client():
    from cryosparc_compute.client import CommandClient
    host = os.environ['CRYOSPARC_MASTER_HOSTNAME']
    port = os.environ['CRYOSPARC_COMMAND_CORE_PORT']
    client = CommandClient(host = host, port = port)
    return client

def wait_job(client, project, job):
    while True:
        sleep(uniform(20, 40))
        job_status = client.get_job_status(project, job)
        if job_status == 'completed':
            return
        elif job_status in ['failed', 'killed']:
            raise RuntimeError(f'Job {project}-{job} terminated with status: {job_status}')

if __name__ == '__main__':
    args = parse_arguments()

    star_path : Path = args['i']
    assert star_path.is_file() and star_path.suffix == '.star', 'Invalid `--input` STAR file'
    star_path = star_path.absolute()
    logger.info(f'Input particle meta path: {str(star_path)}')

    data_dir : Optional[Path] = args['d']
    if data_dir is not None:
        assert data_dir.is_dir(), 'Specified `--directory` does not exist'
        data_dir = data_dir.absolute()
        logger.info(f'Input particle data path: {str(data_dir)}')

    logger.info('Initializing CryoSPARC client')
    client = load_cryosparc_client()
    user = args['user']
    project = args['project']
    workspace = args['workspace']
    lane = args['lane']

    logger.info(f'Logging into CryoSPARC as {user}, working in {project}-{workspace}, using lane {lane}')
    user_id = client.get_id_by_email(user)

    logger.info('Building particle import job')
    import_particle_job = client.make_job(
        user_id = user_id,
        project_uid = project,
        workspace_uid = workspace,
        job_type = 'import_particles',
        params = {
            'particle_meta_path' : str(star_path),
            **({'particle_blob_path' : str(data_dir)} if data_dir is not None else {})
        }
    )
    logger.info(f'Particle import job created with ID: {import_particle_job}')

    logger.info(f'Submitting particle import job {import_particle_job} to queue')
    client.enqueue_job(project, import_particle_job, None, user_id)

    logger.info(f'Waiting for particle import job {import_particle_job} to complete')
    wait_job(client, project, import_particle_job)

    logger.info('Creating 2D classification job')
    class2d_job = client.make_job(
        user_id = user_id,
        project_uid = project,
        workspace_uid = workspace,
        job_type = 'class_2D_new',
        input_group_connects = {
            'particles' : f'{import_particle_job}.imported_particles'
        },
        params = {}
    )
    logger.info(f'2D classification job created with ID: {class2d_job}')

    logger.info(f'Submitting 2D classification job {class2d_job} to queue')
    client.enqueue_job(project, class2d_job, lane, user_id)
