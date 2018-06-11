import numpy as np
from os import listdir, remove
from sys import argv
import multiprocessing

threshold = 20

def fold(file_list, target_file, internal=False):
    n = len(file_list)
    if n < threshold:
        data = np.load(file_list[0])
        if internal:
            print('Removing', file_list[0])
            remove(file_list[0])
        for fpath in file_list[1:]:
            data = np.concatenate((data,np.load(fpath)))
            if internal:
                print('Removing', fpath)
                remove(fpath)
        print('Saving', target_file)
        np.save(target_file, data)
    else:
        pvt = n // 2
        target_file = target_file.rstrip('.npy')
        p = multiprocessing.Process(target=fold,
            args=(file_list[:pvt], target_file+'_0',))
        q = multiprocessing.Process(target=fold,
            args=(file_list[pvt:], target_file+'_1',))
        p.start()
        q.start()
        p.join()
        q.join()
        print('Processed finished')
        fold((target_file+'_0.npy',target_file+'_1.npy',), target_file, internal=True)

if __name__ == '__main__':
    if len(argv) == 4:
        threshold = int(argv[3])
    if len(argv) >= 3:
        src_dir = argv[1]
        target_file = argv[2]
        flist = [src_dir + f for f in listdir(src_dir)]
        fold(flist, target_file)
    else:
        print(('Invalid Argument Count: {}\n'
            'Usage:\n\tpython fold.py <src_dir> <target_file> <?group_size>'.format(len(argv)-1)))
