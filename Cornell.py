import glob
import torch
import numpy as np
from image import Image
from image import GraspImage
from image import DepthImage
from grasp import Grasps
import os
import random
import cv2


class Cornell(torch.utils.data.Dataset):
    # 载入cornell数据集的类
    def __init__(self, file_dir, include_depth=True, include_rgb=True, start=0.0, end=1.0, output_size=300):
        '''
        :功能               : 数据集封装类的初始化函数，功能包括数据集读取，数据集划分，其他参数初始化等
        :参数 file_dir      : str,按照官方文档的示例和之前的经验，这里需要读入数据集，所以需要指定数据的存放路径
        :参数 include_depth : bool,是否包含depth图像输入
        :参数 include_rgb   : bool,是否包含rgb图像输入
        :参数 output_size   : 各图片的输出大小，裁剪得到
        :参数 start,end     : float,为了方便数据集的拆分，这里定义添加两个边界参数start,end
        :返回 None
        '''
        super(Cornell, self).__init__()

        # 一些参数的传递
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.output_size = output_size
        # 去指定路径载入数据集数据

        graspf = glob.glob(os.path.join(file_dir,'*','*os.txt'))
        graspf.sort()

        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('没有查找到数据集，请检查路径{}'.format(file_dir))

        rgbf = [filename.replace('cpos.txt', 'r.png') for filename in graspf]
        depthf = [filename.replace('cpos.txt', 'd.tiff') for filename in graspf]
        grasp_point_f=glob.glob(os.path.join(file_dir, '*', 'pcd*1.png'))
        grasp_point_f.sort()

        # 按照设定的边界参数对数据进行划分并指定为类的属
        self.graspf = graspf[int(l * start):int(l * end)]
        self.rgbf = rgbf[int(l * start):int(l * end)]
        self.depthf = depthf[int(l * start):int(l * end)]
        self.grasp_point_f = grasp_point_f[int(l * start):int(l * end)]
        print('grasp_point_f',len(self.grasp_point_f))
        print('rgbf',len(self.rgbf))
        print('graspf',len(self.graspf))
        print('depthf',len(self.depthf))


    @staticmethod
    def numpy_to_torch(s):
        '''
        :功能     :将输入的numpy数组转化为torch张量，并指定数据类型，如果数据没有channel维度，就给它加上这个维度
        :参数 s   :numpy ndarray,要转换的数组
        :返回     :tensor,转换后的torch张量
        '''
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def _get_crop_attrs(self, idx):
        '''
        :功能     :读取多抓取框中心点的坐标，并结合output_size计算要裁剪的左上角点坐标
        :参数 idx :int,
        :返回     :计算出来的多抓取框中心点坐标和裁剪区域左上角点坐标
        '''
        grasp_rectangles = Grasps.load_from_cornell_files(self.graspf[idx])
        center = grasp_rectangles.center
        # 按照ggcnn里面的话，这里本该加个限制条件，防止角点坐标溢出边界，但前面分析过，加不加区别不大，就不加了
        # print(center)
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top


    def get_rgb(self, idx, rot=0, zoom=1.0):
        rgb_img = Image.from_file(self.rgbf[idx])
        #print(rgb_img.img.shape)
        center, left, top = self._get_crop_attrs(idx)
       # rgb_img.rotate(rot, center.tolist())
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
       # rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        rgb_img.normalize()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = DepthImage.from_file(self.depthf[idx])
        center, left, top = self._get_crop_attrs(idx)
        #depth_img.rotate(rot, center.tolist())
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalize()
        #depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_grasp(self, idx, rot=0, zoom=1.0):
        '''
        :功能       :读取返回指定id的抓取标注参数并将多个抓取框的参数返回融合
        :参数 idx   :int,要读取的数据id
        :参数 pos   :bool,是否生成返回位置映射图
        :参数 angle :bool,是否生成返回角度映射图
        :参数 width :bool,是否生成返回夹爪宽度映射图
        :返回       :以图片的方式返回定义一个抓取的多个参数，包括中心点，角度，宽度和长度
        crop((left, top), (left + self.output_size, top + self.output_size))
        (left, top), (left + self.output_size, top + self.output_size)
        '''
        grasp_img = GraspImage.from_file(self.grasp_point_f[idx])
       # print(grasp_img.img.shape)
        center, left, top = self._get_crop_attrs(idx)
        #grasp_img.rotate(rot, center.tolist())
        grasp_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        #grasp_img.zoom(zoom)
        grasp_img.resize((self.output_size, self.output_size))
       # print(grasp_img.img.shape)
        grasp_map=grasp_img.generate_map()
       # print(grasp_map.shape)
        return grasp_map


    def crop(self, img, top_left, bottom_right):
        '''
        :功能              :按照给定参数对图像进行裁剪操作
        :参数 top_left     :ndarray,要裁剪区域的左上角点坐标
        :参数 bottom_right :ndarray,要裁剪区域的右下角点坐标
        '''
        img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        return img

    def __getitem__(self, idx):
        #data argumentation
        zoom_factor = np.random.uniform(0.5, 1.0)
        rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
        rot = random.choice(rotations)
        depth_img = self.get_depth(idx, rot, zoom_factor)
        rgb_img = self.get_rgb(idx, rot, zoom_factor)
        grasp_img = self.get_grasp(idx, rot, zoom_factor)
        # print(depth_img.shape,rgb_img.shape)
        # print(np.expand_dims(depth_img, 0).shape)
        x = self.numpy_to_torch(
            np.concatenate(
                (np.expand_dims(depth_img, 0),
                 rgb_img),
                0
            )
        )
        # print(x.shape)
        y = self.numpy_to_torch(grasp_img)
        return x, y, idx, rot, zoom_factor

    # 映射类型的数据集，别忘了定义这个函数
    def __len__(self):
        return len(self.graspf)


