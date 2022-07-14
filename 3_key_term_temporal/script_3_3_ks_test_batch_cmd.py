

cmd = '~/anaconda3/envs/sklearn/bin/python script_3_3_ks_test.py'

with open('run_script_3_3_ks_test.sh', 'w') as f:
  f.write('#!/bin/bash --login\n')
  for i in range(0,500,25):
    print(i, i+25)
    f.write(f'{cmd} {i} {i+25} &\n')

