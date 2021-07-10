#!/usr/bin/env python3
"""
filter scan by semantic label, remove moving objects and outliers,
as a points data preprocess for lidar slam.

author: Longrui Dong

"""

import argparse
import os
import yaml
import numpy as np

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("new folder: ", path)
    else:
        print("folder exits !")


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
# labels to remove according to semntic-kitti.yaml
# 1, 252, 253, 254, 255, 256, 257, 258,  filtered-0
rm_label = [1, 10, 13, 18, 30, 252, 253, 254, 255, 256, 257, 258, 259] # imls

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./cloud_filter.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False, # True
        default="/home/dlr/kitti/dataset/",
        help='Dataset to filter. No Default',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="/home/dlr/Project/semantic-kitti-api/config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="04",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default=None,
        required=False,
        help='Alternate location for labels, to use predictions folder. '
        'Must point to directory containing the predictions in the proper format '
        ' (see readme)'
        'Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_semantics', '-i',
        dest='ignore_semantics',
        default=False,
        action='store_true',
        help='Ignore semantics. Visualizes uncolored pointclouds.'
        'Defaults to %(default)s',
    )
    parser.add_argument(
        '--do_instances', '-di',
        dest='do_instances',
        default=False,
        action='store_true',
        help='Visualize instances too. Defaults to %(default)s',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        required=False,
        help='Sequence to start. Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_safety',
        dest='ignore_safety',
        default=False,
        action='store_true',
        help='Normally you want the number of labels and ptcls to be the same,'
        ', but if you are not done inferring this is not the case, so this disables'
        ' that safety.'
        'Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Sequence", FLAGS.sequence)
    print("Predictions", FLAGS.predictions)
    print("ignore_semantics", FLAGS.ignore_semantics)
    print("do_instances", FLAGS.do_instances)
    print("ignore_safety", FLAGS.ignore_safety)
    print("offset", FLAGS.offset)
    print("*" * 80)

    # open config file
    # try:
    #     print("Opening config file %s" % FLAGS.config)
    #     CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    # except Exception as e:
    #     print(e)
    #     print("Error opening yaml file.")
    #     quit()
    
    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))
    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                                FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()
    sequence_length = len(scan_names)
    print( "Sequence {0} Length: {1}".format(FLAGS.sequence, sequence_length) )

    # does sequence folder exist?
    if not FLAGS.ignore_semantics:
        if FLAGS.predictions is not None:
            label_paths = os.path.join(FLAGS.predictions, "sequences",
                                       FLAGS.sequence, "predictions")
        else:
            label_paths = os.path.join(FLAGS.dataset, "sequences",
                                    FLAGS.sequence, "labels")#
        if os.path.isdir(label_paths):
            print("Labels folder exists! Using labels from %s" % label_paths)
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()
        # populate the pointclouds
        label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn]
        label_names.sort()

        # check that there are same amount of labels and scans
        if not FLAGS.ignore_safety:
            assert(len(label_names) == len(scan_names))
        
    # new save paths
    scan_paths_new = os.path.join(FLAGS.dataset, "sequences",
                                FLAGS.sequence, "velodyne_filtered")
    label_paths_new = os.path.join(FLAGS.dataset, "sequences",
                                FLAGS.sequence, "labels_filtered")
    mkdir(scan_paths_new)
    mkdir(label_paths_new)
    # read scan and correspoinding semantic label
    for i in range(sequence_length): #sequence_length
        scanfile_name = scan_names[i] #/home/dlr/kitti/dataset/sequences/04/velodyne/000000.bin
        labelfile_name = label_names[i]#/home/dlr/kitti/dataset/sequences/04/labels/000000.label
        binname = '{0:06d}'.format(int(i))
        scan_name_new = scan_paths_new + "/" + binname + ".bin"
        label_name_new = label_paths_new + "/" + binname + ".label"
        # check filename is string
        if not isinstance(scanfile_name, str):
            raise TypeError("scanfile_name should be string type, "
                            "but was {type}".format(type=str(type(scanfile_name))))
        if not isinstance(labelfile_name, str):
            raise TypeError("labelfile_name should be string type, "
                            "but was {type}".format(type=str(type(labelfile_name))))
        # check extension is a laserscan  .bin
        if not any(scanfile_name.endswith(ext) for ext in EXTENSIONS_SCAN):
            raise RuntimeError("scanfile_name extension is not valid scan file.")
        if not any(labelfile_name.endswith(ext) for ext in EXTENSIONS_LABEL):
            raise RuntimeError("labelfile_name extension is not valid label file.")

        # if all goes well, open pointcloud & label
        scan_i = np.fromfile(scanfile_name, dtype=np.float32)
        # print('raw scan_i shape: ',scan_i.shape)
        scan_i = scan_i.reshape((-1, 4))
        # print('after scan_i reshape: ',scan_i.shape)

        label_i = np.fromfile(labelfile_name, dtype=np.uint32)
        # print('raw label_i shape: ',label_i.shape)
        label_i = label_i.reshape((-1))
        # print('after label_i reshape: ',label_i.shape)

        if scan_i.shape[0] != label_i.shape[0]:
            # print("Points shape: ", scan_i.shape)
            # print("Label shape: ", label_i.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        # initial empty list to store index of moving sths..
        rmid_list = []

        for j in range(label_i.shape[0]):
            sem_label = label_i[j] & 0xFFFF  # semantic label in lower half !!
            # if  any((sem_label == rmind) for rmind in rm_label):
            if (sem_label in rm_label):
                rmid_list.append(j)
        print('need rm ', len(rmid_list))
        # remove corresponding point of those label
        scan_i_rm = np.copy(scan_i)
        scan_i_rm = np.delete(scan_i_rm, rmid_list, axis=0)
        # print("Points shape after filter", scan_i_rm.shape)
        label_i_rm = np.copy(label_i)
        label_i_rm = np.delete(label_i_rm, rmid_list, axis=0)
        if scan_i_rm.shape[0] != label_i_rm.shape[0]:
            # print("Points shape: ", scan_i.shape)
            # print("Label shape: ", label_i.shape)
            raise ValueError("New Scan and Label don't contain same number of points")
        # reshape new scan
        scan_i_new = scan_i_rm.flatten()
        # print("Points shape to save", scan_i_new.shape)
        print("saving new scan & label at :", binname)
        # save as .bin file
        # print("saving new sacn to :", scan_name_new)
        scan_i_new.tofile(scan_name_new)

        #save as .label file label_name_new
        # print("saving new label to :", label_name_new)
        label_i_rm.tofile(label_name_new)






