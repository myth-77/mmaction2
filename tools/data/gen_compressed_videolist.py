import mythDecoder as md
from tqdm import tqdm
import argparse
import os
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reading compressed video to generate video list')
    parser.add_argument(
        '--dataroot',
        type=str,
        help='dataset root')
    parser.add_argument(
        '--format',
        type=int,
        choices=[4,264,265]
    )
    parser.add_argument(
        '--datalistdir',
        type=str
    )
    parser.add_argument(
        '--outdir',
        type=str
    )
    parser.add_argument(
        '--subfix',
        default='mp4',
        type=str
    )
    args = parser.parse_args()
    print(f'datalist dir {args.datalistdir}')
    datalists = os.listdir(args.datalistdir)
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    
    info_d = dict()
    write_files =defaultdict(list)
    for datalist in datalists:
        
        datalist_path = os.path.join(args.datalistdir, datalist)
        print(f'reading {datalist_path}')
        if not os.path.isfile(datalist_path):
            continue
        with open(datalist_path) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                video_sub_path, cls_name, cls_id = line.strip().split()
                video_sub_path = os.path.splitext(video_sub_path)[0]+'.'+args.subfix
                if not video_sub_path in info_d.keys():
                    num_frames, num_gop = md.get_num_frames(os.path.join(args.dataroot, video_sub_path), args.format)
                    info_d[video_sub_path] = (cls_id, num_frames, num_gop)
                write_files[datalist].append(video_sub_path)
    
    for write_file in write_files.keys():
        
        datalist_path = os.path.join(args.outdir, write_file)
        print(f'writing {datalist_path}')
        with open(datalist_path, 'w') as f:
            for video_sub_path in write_files[write_file]:
                f.write(f'{video_sub_path} {info_d[video_sub_path][0]} {info_d[video_sub_path][1]} {info_d[video_sub_path][2]}\n')

    