import os
import pickle
import csv
input_data_dir = '/BS/discrete_opt/nobackup/miplib_collection/lp_format_presolved/'
out_dir = '/home/ahabbas/data/learnDBCA/miplib'
data_csv = csv.reader(open('/BS/discrete_opt/nobackup/miplib_collection/the_binary_tag.csv', 'r'))
data_names = []
data_types = []
for (i, line) in enumerate(data_csv):
    if i == 0:
        continue
    data_names.append(line[0])
    data_types.append(line[1])

for (name, typ) in zip(data_names, data_types):
    for disk_name in os.listdir(input_data_dir):
        if name in disk_name:
            src = os.path.join(input_data_dir, disk_name)
            dst = os.path.join(out_dir, typ, 'instances', disk_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if (os.path.isfile(dst)):
                continue
            os.symlink(src, dst, target_is_directory=False)
            print(f'Created: {src}, {dst}')
