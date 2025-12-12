# -*- coding: utf-8 -*-
"""
Created on 18.07.2024
@author: eschlager
Read yaml file with model and training specifications
"""

import yaml
import logging

def read_yaml_file(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            specs = yaml.safe_load(stream)
            logging.info(f'Read specification file {yaml_file} successfully.')
            return specs
        except yaml.YAMLError as exc:
            logging.exception(exc)    
