#!/usr/bin/env python
import argparse
import boto3
import os
import json
import time
import webbrowser


def get_instance_ip_address(instance_info):
    return instance_info['PublicIpAddress']


def get_instance_info(ec2, instance_id):
    response = ec2.describe_instances(InstanceIds=[instance_id])
    return response['Reservations'][0]['Instances'][0]


def get_instance_state(instance_info):
    return instance_info['State']['Name']


def stop_instance(ec2, instance_id):
    ec2.stop_instances(InstanceIds=[instance_id])


def start_instance(ec2, instance_id):
    ec2.start_instances(InstanceIds=[instance_id])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['start', 'stop', 'ssh', 'show'])
    parser.add_argument('--browser', '-B', action='store_true')
    opts = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), 'credentials.json')) as f:
        creds = json.load(f)

    ec2 = boto3.client('ec2', aws_access_key_id=creds['aws_access'],
                       aws_secret_access_key=creds['aws_secret'],
                       region_name='us-east-2')
    inst_id = creds['instance_id']

    if opts.command == 'start':
        start_instance(ec2, inst_id)
    elif opts.command == 'show':
        info = get_instance_info(ec2, inst_id)
        ip_addr = get_instance_ip_address(info)
        url = 'http://{}:9999'.format(ip_addr)
        print(url)
        if opts.browser:
            webbrowser.open(url)
    elif opts.command == 'stop':
        stop_instance(ec2, inst_id)
    elif opts.command == 'ssh':
        while True:
            info = get_instance_info(ec2, inst_id)
            state = get_instance_state(info)
            if state == 'pending':
                print('Instance is in pending state, wait...')
                time.sleep(5)
            elif state == 'running':
                break
            else:
                raise Exception(state)
        ip_addr = get_instance_ip_address(info)
        print('Got address of instance:', ip_addr)
        flags = []
        if creds.get('cert_path'):
            flags.extend(('-i', creds['cert_path']))
        flag_str = ' '.join(flags)
        os.system('ssh {} ubuntu@{}'.format(flag_str, ip_addr))
    else:
        raise NotImplementedError(opts.command)


if __name__ == '__main__':
    main()
