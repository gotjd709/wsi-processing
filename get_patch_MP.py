import multiprocessing
import argparse
import util_multi
import glob

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--slide_path', required=True, type=str, help='Input the target slide path.')
parser.add_argument('--classes', required=True, type=int, choices=[3, 4], help='Input the class of the mask.')
parser.add_argument('--level', required=True, type=int, choices=[1, 2], help='Input the target level of the annotation mask to be referenced.')
parser.add_argument('--anno_percent', required=True, type=float, help='Input the minimum percentage of annotations in a patch.')
parser.add_argument('--patch_size', required=True, type=int, choices=[284, 512], help='Input the image shape.')
parser.add_argument('--magnification',required=True, type=int, choices=[50, 100, 200],help='Input the target magnification of the patch.')
parser.add_argument('--save_patch_path', required=True, type=str, help='Input the patch path that will be saved.')

# argparse to variable
args = parser.parse_args()
slide_path = args.slide_path
classes = args.classes
level = args.level
anno_percent = args.anno_percent
patch_size = args.patch_size
magnification = args.magnification
save_patch_path = args.save_patch_path
hooknet = None if patch_size == 512 else True

# init_params setting
init_params = {
    'svs_path' : '',
    'xml_path' : '',
    'level' : level,
    'patch_size' : patch_size,
    'save_patch_path' : save_patch_path            
}

# svsfile_to_patch extrcator
def extract(init_params, magnification=magnification):
    slide = util_multi.processor(init_params)
    slide.get_patch(magnification=magnification, anno_percent=anno_percent, classes=classes, hooknet=hooknet)

# slide_alth and list setting
slide_path = slide_path + '*.svs' 
svs_list = glob.glob(slide_path)

# using multiprocess 
for svs_path in svs_list:
    xml_path = '.'.join(svs_path.split('.')[:-1]) + '.xml'
    init_params.update({'svs_path':svs_path, 'xml_path':xml_path})
    p = multiprocessing.Process(target=extract, args=(init_params,))
    p.start()