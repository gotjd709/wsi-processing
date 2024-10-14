from skimage             import io, morphology, measure, filters
from lxml                import etree
from tqdm                import tqdm, trange
import matplotlib.pyplot as plt
import numpy             as np 
import openslide
import cv2
import os

'''
init_param= {
    'svs_path': ,
    'xml_path': ,
    'level': ,
    'patch_size' :
    'save_patch_path' : 
}
'''

class processor(object):

    # init variable setting method
    def __init__(self, init_param):
        self.slide_path = init_param['svs_path']
        self.annotation = '' if init_param['xml_path'] == '' else etree.parse(init_param['xml_path'])
        self.level = init_param['level']
        self.patch_size = init_param['patch_size']
        self.save_patch_path = init_param['save_patch_path']
        
        self.slide = openslide.OpenSlide(self.slide_path)
        self.properties = self.slide.properties
        self.src_w, self.src_h = self.slide.level_dimensions[0]
        self.dest_w, self.dest_h = self.slide.level_dimensions[self.level]
        self.multiple = self.src_w // self.dest_w
        self.mpp = float(self.slide.properties.get('openslide.mpp-x'))
        self.arr = np.array(self.slide.read_region((0,0), self.level, size=(self.dest_w,self.dest_h)).convert('RGB'))


    # get thumnail method
    def get_thumbnail(self, show=False):
        self.slide.get_thumbnail(self.slide.level_dimensions[self.level])
        if show == True:
            plt.imshow(self.slide.get_thumbnail(self.slide.level_dimensions[self.level]))


    # visualize all annotation method
    def visualize_all_annotation(self):
        root = self.annotation.getroot()
        sub_root = root[0]
        total_points_org = []
        total_points = []

        WSI_array = np.array(self.slide.read_region([10, 10], self.level, [self.dest_w, self.dest_h]))[:,:,0:3]
        WSI_array = cv2.cvtColor(WSI_array, cv2.COLOR_BGR2RGB)
        mask = np.zeros((self.dest_h, self.dest_w), dtype=np.uint8)

        for i in range(len(sub_root)):
            xy_point_org = []
            xy_point = []

            Type = sub_root[i].attrib['Type']
            Name = sub_root[i].attrib['Name']
            PartOfGroup = sub_root[i].attrib['PartOfGroup']

            ### tumor boundary => Red
            if Type == 'Rectangle' and Name == 'tumor_boundary':
                for child in sub_root[i].iter('Coordinate'):
                    coordinates = child.attrib
                    x = (float(coordinates.get('X')))
                    y = (float(coordinates.get('Y')))
                    xy_point_org.append((x,y))
                    xy_point.append((x/self.multiple, y/self.multiple))
                total_points_org.append(xy_point_org)
                total_points.append(xy_point)
                

                mask_pts = np.array(xy_point).reshape((-1,1,2)).astype(np.int32)
                cv2.drawContours(WSI_array, [mask_pts], -1, (255, 0, 0), 10)
            
            ### vertical boundary => Blue 
            if Type == 'Rectangle' and Name == 'vertical_boundary':
                for child in sub_root[i].iter('Coordinate'):
                    coordinates = child.attrib
                    x = (float(coordinates.get('X')))
                    y = (float(coordinates.get('Y')))
                    xy_point_org.append((x,y))
                    xy_point.append((x/self.multiple, y/self.multiple))
                total_points_org.append(xy_point_org)
                total_points.append(xy_point)

                mask_pts = np.array(xy_point).reshape((-1,1,2)).astype(np.int32)
                cv2.drawContours(WSI_array, [mask_pts], -1, (0, 0, 255), 10)

            ### spline (annotation) Normal => Green
            if Type == 'Spline' and PartOfGroup[0] == '1':
                for child in sub_root[i].iter('Coordinate'):
                    coordinates = child.attrib
                    x = (float(coordinates.get('X')))
                    y = (float(coordinates.get('Y')))
                    xy_point_org.append((x,y))
                    xy_point.append((x/self.multiple, y/self.multiple))
                total_points_org.append(xy_point_org)
                total_points.append(xy_point)

                mask_pts = np.array(xy_point).reshape((-1,1,2)).astype(np.int32)
                cv2.drawContours(WSI_array, [mask_pts], -1, (0, 255, 0), 10)

            ### spline (annotation) Tumor => Yellow
            if Type == 'Spline' and PartOfGroup[0] == '0':
                for child in sub_root[i].iter('Coordinate'):
                    coordinates = child.attrib
                    x = (float(coordinates.get('X')))
                    y = (float(coordinates.get('Y')))
                    xy_point_org.append((x,y))
                    xy_point.append((x/self.multiple, y/self.multiple))
                total_points_org.append(xy_point_org)
                total_points.append(xy_point)

                mask_pts = np.array(xy_point).reshape((-1,1,2)).astype(np.int32)
                cv2.drawContours(WSI_array, [mask_pts], -1, (255, 255, 0), 10)
            
        plt.figure(figsize=(15, 15))
        plt.imshow(WSI_array)
        plt.show()


    # for annotation bbox min max restriction 
    def min_max_restriction(self, input_num, min_num, max_num):
        if input_num < min_num:
            return min_num
        elif input_num > max_num:
            return max_num
        else:
            return input_num  


    # get asap annotation dictionary method
    def get_annotation_asap(self):
        total = list()
        bbox_list = list()
        anno_dict = dict()
        trees = self.annotation.getroot()[0]
        for tree in trees:

            ## bbox coordinates extraction
            if str(tree.get('PartOfGroup').split('_')[0]) == 'None':
                group = 'boundary'
                regions = tree.findall('Coordinates')
                for region in regions:
                    coordinates = region.findall('Coordinate')
                    bbox = list()
                    for coord in coordinates:
                        x = round(float(coord.get('X')))
                        y = round(float(coord.get('Y')))
                        x = np.clip(round(x/self.multiple), 0, self.dest_w)
                        y = np.clip(round(y/self.multiple), 0, self.dest_h)
                        bbox.append((x, y))
                    bbox_list.append(bbox)

            ## get annotation restricted coordinates
            else:
                if tree.get('PartOfGroup').split('_')[0][:1] == '0':
                    group = 'p_' + str(tree.get('PartOfGroup').split('_')[0])
                    regions = tree.findall('Coordinates')
                    for region in regions:
                        coordinates = region.findall('Coordinate')
                        pts = list()
                        for coord in coordinates:
                            x = round(float(coord.get('X')))
                            y = round(float(coord.get('Y')))
                            x = np.clip(round(x/self.multiple), 0, self.dest_w)
                            y = np.clip(round(y/self.multiple), 0, self.dest_h)
                            pts.append((x, y))
                        if group in anno_dict.keys():
                            anno_dict[group].append(pts)
                        else:
                            anno_dict[group] = [pts]

        total.append(bbox_list)
        total.append(anno_dict)

        return total
    
    
    def mask_tone(self, key):
        if key == 'p_01' or key == 'p_02' or key == 'p_06': 
            return 1
        elif key == 'p_03' or key == 'p_04' or key == 'p_05' or key == 'p_07' or key == 'p_08':
            return 2

    # get asap annotation mask method
    def get_anno_mask(self, show=False):
        total = list()
        bbox_list = self.get_annotation_asap()[0]
        anno_dict = self.get_annotation_asap()[1]
        mask_dict = dict()
        total.append(bbox_list)

        for key in anno_dict.keys():
            mask = np.zeros((self.dest_h, self.dest_w))
            regions = anno_dict[key]
            for region in regions:
                pts = [np.array(region, dtype=np.int32)]
                mask = cv2.fillPoly(mask, pts, self.mask_tone(key))
            mask_dict[key] = mask

        total.append(mask_dict)

        if show==True:
            plt.figure(figsize=(15, 15*len(mask_dict.keys())//2))
            sum_np = np.zeros((self.dest_h, self.dest_w))
            for i, patch in enumerate(mask_dict.keys()):
                sum_np = np.add(sum_np, mask_dict[patch])
                unique, counts = np.unique(mask_dict[patch], return_counts = True)
                uniq_cnt_dict = dict(zip(unique, counts))
                print(f'mask element count : {uniq_cnt_dict}', '\n')
                plt.subplot(len(mask_dict.keys())+1, 1, i+1)
                plt.imshow(mask_dict[patch], vmin=0, vmax=255)
                plt.title(patch)
            unique, counts = np.unique(sum_np, return_counts = True)
            uniq_cnt_dict = dict(zip(unique, counts))
            print(f'sum mask element count : {uniq_cnt_dict}', '\n')
            plt.subplot(len(mask_dict.keys())+1, 1, i+2)
            plt.imshow(sum_np, vmin=0, vmax=255)
            plt.title('All_Mask_Sum')

        return total


    # get tissue mask method
    def get_tissue_mask(self, area_thr=1000000, RGB_min=0, show=False):

        hsv = cv2.cvtColor(self.arr, cv2.COLOR_RGB2HSV)
        ## if more than threshold make Ture
        background_R = self.arr[:, :, 0] > filters.threshold_otsu(self.arr[:, :, 0])
        background_G = self.arr[:, :, 1] > filters.threshold_otsu(self.arr[:, :, 1])
        background_B = self.arr[:, :, 2] > filters.threshold_otsu(self.arr[:, :, 2])

        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])

        min_R = self.arr[:, :, 0] > RGB_min
        min_G = self.arr[:, :, 1] > RGB_min
        min_B = self.arr[:, :, 2] > RGB_min

        mask = tissue_S & (tissue_RGB + min_R + min_G + min_B)
        ret = morphology.remove_small_holes(mask, area_threshold=area_thr)
        ret = np.array(ret).astype(np.uint8)
        
        tissue_mask = cv2.morphologyEx(ret*255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))) 

        if show ==True:
            plt.imshow(ret)    

        return tissue_mask

    # get sequence ratio method
    def get_seq_range(self, slide_width, slide_height, multiple, img_size, s):
        y_seq = trange(int(((slide_height*multiple) - int(img_size)) // int(s) + 1))
        x_seq = range(int(((slide_width*multiple) - int(img_size)) // int(s) + 1))
        return y_seq, x_seq

    # get ratio mask method
    def get_ratio_mask(self, patch):
        h_, w_ = patch.shape[0], patch.shape[1]
        n_total = h_*w_
        n_cell = np.count_nonzero(patch)
        if (n_cell != 0):
            return n_cell*1.0/n_total*1.0
        else:
            return 0


    # save patch method
    def save_patch(self, dir_path, file_name, img):
        os.makedirs(dir_path, exist_ok = True)
        cv2.imwrite(os.path.join(dir_path, file_name), img)


    # extract patch from get patch 
    def execute_patch(self, img_patch1, img_patch2, img_patch3, img_patch4, mask_img, patch_count, save_dir, name, hooknet):
        resize_image1 = cv2.resize(img_patch1, (self.patch_size, self.patch_size), cv2.INTER_AREA)
        resize_image2 = cv2.resize(img_patch2, (self.patch_size, self.patch_size), cv2.INTER_AREA)
        resize_image3 = cv2.resize(img_patch3, (self.patch_size, self.patch_size), cv2.INTER_AREA)
        resize_image4 = cv2.resize(img_patch4, (self.patch_size, self.patch_size), cv2.INTER_AREA)
        resize_mask = cv2.resize(mask_img, (int(self.patch_size-214), int(self.patch_size-214)), cv2.INTER_LINEAR) if hooknet else cv2.resize(mask_img, (int(self.patch_size), int(self.patch_size)), cv2.INTER_LINEAR)
        
        self.save_patch(save_dir + '/' + self.slide_path.split('.')[-2][-3:] + '/input_x1', f'{patch_count}{name}.png', resize_image1)
        self.save_patch(save_dir + '/' + self.slide_path.split('.')[-2][-3:] + '/input_x2', f'{patch_count}{name}.png', resize_image2)
        self.save_patch(save_dir + '/' + self.slide_path.split('.')[-2][-3:] + '/input_x4', f'{patch_count}{name}.png', resize_image3)
        self.save_patch(save_dir + '/' + self.slide_path.split('.')[-2][-3:] + '/input_x8', f'{patch_count}{name}.png', resize_image4)
        self.save_patch(save_dir + '/' + self.slide_path.split('.')[-2][-3:] + '/input_y1', f'{patch_count}{name}.png', resize_mask)


    # get slide patch corresponding with annoation mask method
    def get_patch(self, magnification, anno_percent, classes, hooknet):
        # check slide name
        slide_name = self.slide_path.split('/')[-1][:-4]
        print(slide_name)

        # mask setting and mpp setting
        tissue_mask = self.get_tissue_mask(area_thr=1000000)
        bbox_list = self.get_anno_mask()[0]
        anno_mask = self.get_anno_mask()[1]
        origin_magnification = float(self.properties['openslide.objective-power'])*10
        devided_magnification = origin_magnification // magnification

        # patch size setting
        if hooknet:
            print('hooknet mode!')
            patch_size_lv0 = int(284 * devided_magnification)
            mask_size_lv0 = int(70 * devided_magnification)        
        else:
            print('normal mode!')
            patch_size_lv0 = int(self.patch_size * devided_magnification)

        # patch generation setting
        step = 1
        patch_count = 0
        save_patch = self.save_patch_path

        # loading bounding boxes one by one 
        for bbox in bbox_list:
            min_x = min(j[0] for j in bbox)
            max_x = max(j[0] for j in bbox)
            min_y = min(j[1] for j in bbox)
            max_y = max(j[1] for j in bbox)

            # patch range generation
            slide_w = max_x - min_x
            slide_h = max_y - min_y
            y_seq, x_seq = self.get_seq_range(slide_w, slide_h, self.multiple, patch_size_lv0, patch_size_lv0)

            # loading patch on by one
            for y in y_seq:
                for x in x_seq:

                    # patch location setting
                    start_x = int(min_x + (mask_size_lv0*x/self.multiple)) if hooknet else int(min_x + (patch_size_lv0*x/self.multiple))
                    end_x = int(min_x+(mask_size_lv0*(x+step)/self.multiple)) if hooknet else int(min_x+(patch_size_lv0*(x+step)/self.multiple))
                    start_y = int(min_y + (mask_size_lv0*y/self.multiple)) if hooknet else int(min_y + (patch_size_lv0*y/self.multiple))
                    end_y = int(min_y+(mask_size_lv0*(y+step)/self.multiple)) if hooknet else int(min_y+(patch_size_lv0*(y+step)/self.multiple))                    

                    img1_x_start = int(start_x*self.multiple - 107*devided_magnification) if hooknet else int(start_x*self.multiple)
                    img1_y_start = int(start_y*self.multiple - 107*devided_magnification) if hooknet else int(start_y*self.multiple)
                    img2_x_start = int(start_x*self.multiple - 107*devided_magnification - patch_size_lv0*(1/2)) if hooknet else int(start_x*self.multiple - patch_size_lv0*(1/2))
                    img2_y_start = int(start_y*self.multiple - 107*devided_magnification - patch_size_lv0*(1/2)) if hooknet else int(start_y*self.multiple - patch_size_lv0*(1/2))
                    img3_x_start = int(start_x*self.multiple - 107*devided_magnification - patch_size_lv0*(3/2)) if hooknet else int(start_x*self.multiple - patch_size_lv0*(3/2))
                    img3_y_start = int(start_y*self.multiple - 107*devided_magnification - patch_size_lv0*(3/2)) if hooknet else int(start_y*self.multiple - patch_size_lv0*(3/2))
                    img4_x_start = int(start_x*self.multiple - 107*devided_magnification - patch_size_lv0*(7/2)) if hooknet else int(start_x*self.multiple - patch_size_lv0*(7/2))
                    img4_y_start = int(start_y*self.multiple - 107*devided_magnification - patch_size_lv0*(7/2)) if hooknet else int(start_y*self.multiple - patch_size_lv0*(7/2))

                    # generating 1x magnification of mask
                    img_patch1 = np.array(self.slide.read_region(
                        location = (img1_x_start, img1_y_start),
                        level = 0,
                        size = (patch_size_lv0, patch_size_lv0)
                    )).astype(np.uint8)[...,:3]
                
                    # generating 2x magnification of mask
                    img_patch2 = np.array(self.slide.read_region(
                        location = (img2_x_start, img2_y_start),
                        level = 0,
                        size = (patch_size_lv0*2, patch_size_lv0*2)
                    )).astype(np.uint8)[...,:3]

                    # generating 4x magnification of mask
                    img_patch3 = np.array(self.slide.read_region(
                        location = (img3_x_start, img3_y_start),
                        level = 0,
                        size = (patch_size_lv0*4, patch_size_lv0*4)
                    )).astype(np.uint8)[...,:3]

                    # generating 8x magnification of mask
                    img_patch4 = np.array(self.slide.read_region(
                        location = (img4_x_start, img4_y_start),
                        level = 0,
                        size = (patch_size_lv0*8, patch_size_lv0*8)
                    )).astype(np.uint8)[...,:3]

                    # generating a tissue patch
                    tissue_mask_patch = tissue_mask[start_y:end_y, start_x:end_x]
                    sum_patch = np.zeros(((end_y - start_y), (end_x - start_x)))
                    name = ''
                    
                    # extracting annotation from annotation patch and naming patch
                    for anno_mask_key in anno_mask.keys():
                        anno_patch = anno_mask[anno_mask_key][start_y:end_y, start_x:end_x]
                        if (self.get_ratio_mask(anno_patch) >= anno_percent) and (self.get_ratio_mask(tissue_mask_patch) >= 0.3):
                            and_patch = np.logical_and(anno_patch, tissue_mask_patch)*self.mask_tone(anno_mask_key)
                            sum_patch = np.subtract(np.add(and_patch, sum_patch), np.logical_and(and_patch, sum_patch)*self.mask_tone(anno_mask_key))
                            name += '_' + anno_mask_key

                    # saving image patches and a mask patch
                    ## 3 classes mask generation (0: background, 1: tumor1, 2: tumor2)
                    if classes == 3:
                        sum_patch = sum_patch.astype(np.uint8)
                        if sum_patch.any() != 0:
                            print('3 classes patch generation')
                            patch_count += 1 
                            self.execute_patch(img_patch1, img_patch2, img_patch3, img_patch4, sum_patch, patch_count, save_dir=save_patch, name=name, hooknet=hooknet)
                    ## 4 classes mask generation (0: background, 1: tissue, 2: tumor1, 3: tumor2)
                    elif classes == 4:    
                        normal = np.logical_or(tissue_mask_patch,sum_patch)*1
                        sum_patch = np.add(normal,sum_patch).astype(np.uint8)
                        if name == '':
                            name = '_p_10'  # naming tissue patch
                        if self.get_ratio_mask(sum_patch) >= 0.3:
                            print('4 classes patch generation')
                            patch_count += 1 
                            self.execute_patch(img_patch1, img_patch2, img_patch3, img_patch4, sum_patch, patch_count, save_dir=save_patch, name=name, hooknet=hooknet)

 