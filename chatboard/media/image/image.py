import os
from typing import List
from PIL import UnidentifiedImageError
import numpy as np
import base64
import io
from pydantic import BaseModel

import requests
# from config import DEBUG, TEMP_DIR
from util.boto import get_s3_obj, upload_s3_obj
import pathlib
import os

DEBUG = os.environ.get('DEBUG', False)



# https://medium.com/@ajeet214/image-type-conversion-jpg-png-jpg-webp-png-webp-with-python-7d5df09394c9
# try:
#     import cv2
#     import PIL.Image as pimg
# except ModuleNotFoundError:
#     pimg = None
#     print("cv2 and PIL are not installed. Image class will not work.")
#     pass
import cv2
import PIL.Image as pimg

if DEBUG:
    try:
        import matplotlib.pyplot as plt
    except:
        pass


class BoundingBox(BaseModel):
    point1: tuple
    point2: tuple


class ImageError(Exception):
    pass

def fetch_url(url, verify):
    headers = {'User-Agent': 'Musbot/0.1'}
    response = requests.get(url, headers=headers, verify=verify, stream=True)
    if response.status_code != 200:
        raise Exception(f'image fetching is not succesfull: {response.reason}')    
    
    pil_img =pimg.open(io.BytesIO(response.raw.read()))
    # except UnidentifiedImageError as e:
    #     with tempfile.NamedTemporaryFile(suffix=f".jpg", dir=TEMP_DIR) as f:
    #         f.write(response.content)
    #         pil_img = pimg.open(f.name)
    img = Image.from_pil(pil_img)
    img.ext = get_extention(url)
    return img
    # ext = get_extention(url)
    # with tempfile.NamedTemporaryFile(suffix=f".{ext}", dir=TEMP_DIR) as f:
    #     f.write(response.content)
    #     im = Image.from_file(f.name)        
    #     return im
    


def image_grid(imgs, rows, cols, opacity=True, title=None):
    # assert len(imgs) == rows*cols
    
    w, h = imgs[0].size
    colors = 'RGBA' if opacity else 'RGB'
    grid = pimg.new(colors, size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img.pil_img, box=(i%cols*w, i//cols*h))
    return grid

def get_extention(filename: str) -> str:
    # ext = filename.split('.')[-1]
    _, ext = os.path.splitext(filename)
    ext = ext[1:]
    if ext.lower() == 'jpg':
        ext = 'jpeg'
    return ext

class Image:

    def __init__(self, opacity=True) -> None:
        self._np_arr = None
        self._pil_img = None
        self.opacity = opacity
        self.ext = 'png'
        self._height = None
        self._width = None

    
    @property
    def size(self) -> tuple:
        # ! [width, height]
        return self.pil_img.size

    @property
    def shape(self) -> tuple: 
        # ! [height, width, channels]
        return self.np_arr.shape
    
    @property
    def width(self) -> int:
        if self._width is None:
            self._width = self.size[0]
        return self._width
    
    @width.setter
    def width(self, value: int) -> None:
        self._width = value
    
    @property
    def height(self) -> int:
        if self._height is None:
            self._height = self.size[1]
        return self._height
    
    @height.setter
    def height(self, value: int) -> None:
        self._height = value

    @property
    def np_arr(self) -> np.ndarray:
        if self._np_arr is None:
            self._pil_to_numpy()
        return self._np_arr

    @property
    def pil_img(self) -> pimg:
        if self._pil_img == None:
            self._numpy_to_pil()
        return self._pil_img

    @np_arr.setter
    def np_arr(self, value: np.ndarray) -> None:
        self._np_arr = value

    @pil_img.setter
    def pil_img(self, value: pimg) -> None:
        self._pil_img = value

    @staticmethod
    def from_numpy(image: np.ndarray) -> 'Image':
        opacity = len(image.shape) == 3 and image.shape[2] == 4
        img = Image(opacity=opacity)
        img.np_arr = image if image.dtype == np.uint8 else image.astype(np.uint8)
        return img

    @staticmethod
    def from_pil(image: pimg) -> 'Image':
        img = Image()
        img.pil_img = image.copy()
        # img.pil_img.convert('RGB')
        return img

    @staticmethod
    def from_data_url(data_url: str, rgb=False) -> 'Image':
        if 'data:image' in data_url:
            _, data_url = data_url.split(',')
        data = base64.b64decode(data_url)
        img = Image()
        img.pil_img = pimg.open(io.BytesIO(data))
        if rgb:
            img.pil_img = img.pil_img.convert('RGB')
        return img
    
    @staticmethod
    def from_bytes(data: bytes) -> 'Image':
        if isinstance(data, io.BytesIO):
            data = data.read()
        img = Image()
        img.pil_img = pimg.open(io.BytesIO(data))
        return img
    

    @staticmethod
    def from_file(file_path: str) -> 'Image':
        img = Image()
        img.pil_img = pimg.open(file_path)
        img.ext = get_extention(file_path)
        return img

    @staticmethod
    def from_url(url: str, verify: bool=None) -> 'Image':
        # response = requests.get(url, verify=verify)      
        # pil_img = pimg.open(io.BytesIO(response.content))
        # return Image.from_pil(pil_img)
        return fetch_url(url, verify=verify)
    
        

    @staticmethod
    def from_values(value: int, width: int, height: int) -> 'Image':
        img = Image()
        img.np_arr = np.full((height, width), value)
        return img
    
    @staticmethod
    def plain_color(width, height, color=[0,0,0]):
        color = np.array(color)
        np_arr = np.full((height, width, 3), color, dtype=np.uint8)
        return Image.from_numpy(np_arr)

    def to_stream(self, ext='png', quality=100) -> io.BytesIO:
        img_io = io.BytesIO()
        self.pil_img.save(img_io, ext, quality=quality)
        img_io.seek(0)
        return img_io

    def to_file(self, file_path: str) -> None:
        if type(file_path) == pathlib.PosixPath:
            file_path = str(file_path)
        file_ext = get_extention(file_path)
        if file_ext != self.ext:
            self.pil_img.convert("RGB").save(file_path, file_ext)
        self.pil_img.save(file_path)


    def to_s3(self, bucket: str, file_name: str) -> None:
        img_io = self.to_stream(get_extention(file_name))
        # self.pil_img.save(img_io, 'JPEG', quality=70)
        # self.pil_img.save(img_io, 'JPEG')
        img_io.seek(0)
        return upload_s3_obj(bucket, file_name, img_io)
    
    
    @staticmethod
    def from_s3( bucket: str, file_name: str):
        try:
            img_io = get_s3_obj(bucket, file_name)
            return Image.from_bytes(img_io.read())
        except UnidentifiedImageError as e:
            raise ImageError(f"Image is not valid: {file_name}")

        
        # return 'https://moviemaker-images-public.s3.eu-west-2.amazonaws.com/' + file_name
    def get_thumbnail(self, thum_height=150):
        thum_height = 150
        thum_width = int(self.width * thum_height / self.height)
        thunm_arr = cv2.resize(self.np_arr, (thum_width, thum_height))
        return Image.from_numpy(thunm_arr)

    def invert(self) -> 'Image':
        # img = Image.from_pil(self.pil_img)
        # invert = cv2.bitwise_not(self.np_arr)
        tmp_arr = self.np_arr.copy()
        tmp_arr[:,:,0:3] = 255 - tmp_arr[:,:,0:3]
        return Image.from_numpy(tmp_arr)
        # return Image.from_numpy(255 - self.np_arr)
        # return Image.from_numpy(255 - self.np_arr)

    def opacity_to_mask(self):
        shape = self.np_arr.shape
        tmp_arr = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        opacity_idx = self.np_arr[:,:,3] != 0
        tmp_arr[opacity_idx] = 255
        img = Image.from_numpy(tmp_arr)
        return img
    
    def mask_cut(self, mask: 'Image'):
        # mask = mask.opacity_to_mask()
        # mask = mask.invert()
        if mask.size != self.size:
            mask = mask.resize(self.size)
        img = Image.from_numpy(cv2.bitwise_and(self.np_arr, mask.np_arr))
        return img
    
    def mask_condition(self, condition: np.ndarray):
        img = Image.from_numpy(np.where(condition, self.np_arr, 0))
        return img
    
    def crop(self, x: int, y: int, width: int, height: int) -> 'Image':
        return Image.from_pil(self.pil_img.crop((x, y, x + width, y + height)))
    
    def mask_blur(self, mask: 'Image'):
        if mask.size != self.size:
            mask = mask.resize(self.size)
        # image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)
        image_data = self.np_arr
        # Apply effects
        blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
        # condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        condition = mask.np_arr > 0.1
        output_image = np.where(condition, image_data, blurred_image)
        return Image.from_numpy(output_image)
    
    def gray(self):
        gray = cv2.cvtColor(self.np_arr, cv2.COLOR_BGR2GRAY)
        return Image.from_numpy(gray)

    
    def contours(self, color=(255, 255, 255), thickness=2):
        gray = self.gray().np_arr
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh = cv2.threshold(gray, 1, 255,  cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline_image = np.zeros_like(self.np_arr)
        contour_image = cv2.drawContours(outline_image, contours, -1, color, thickness)
        return Image.from_numpy(contour_image)
    
    def size_in_bytes(self):
        return self.to_stream().getbuffer().nbytes
    

    def draw_bounding_boxs(self, boxs: List[BoundingBox]):
        # pimg = self.pil_img.copy()
        # for box in box:
        #     draw = pimg.Draw(pimg)
        #     draw.rectangle(box, outline='red')
        if type(boxs) != list:
            boxs = [boxs]
        
        arr = self.np_arr.copy()
        width = self.width
        height = self.height
        for box in boxs:
            cv2.rectangle(arr, (int(box.point1[0] * width), int(box.point1[1] * height)), (int(box.point2[0] * width), int(box.point2[1] * height)), (0, 255, 0), 2)
        return Image.from_numpy(arr)
    

    def to_data_url(self):
        img_io = self.to_stream()
        return 'data:image/png;base64,' + base64.b64encode(img_io.getvalue()).decode('ascii')

    def to_base64(self, format='JPEG'):
        # img_io = self.to_stream()
        img_io = io.BytesIO()
        self.pil_img.save(img_io, format='JPEG')
        base64_image = base64.b64encode(img_io.getvalue()).decode('utf-8')
        return base64_image

    def add_alpha(self):
        # image_with_alpha = cv2.cvtColor(self.np_arr, cv2.COLOR_BGR2BGRA)
        image_with_alpha = cv2.cvtColor(self.np_arr, cv2.COLOR_RGB2RGBA)
        image_with_alpha[:, :, 3] = 255
        return Image.from_numpy(image_with_alpha)
    
    def transperent(self):
        # if not self.opacity:
        #     np_arr = self.add_alpha().np_arr
        # else:
        #     np_arr = self.np_arr
        # r, g, b, a = cv2.split(np_arr)

        r, g, b = cv2.split(self.np_arr)
        # Create a mask by thresholding the RGB channels to identify black pixels
        mask = cv2.bitwise_or(cv2.bitwise_or(r, g), b)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Calculate the alpha channel based on the intensity of the black pixels
        alpha = cv2.subtract(255, mask)

        # Merge the modified channels back into a 4-channel image
        modified_image = cv2.merge((b, g, r, alpha))



        # # Create a mask by thresholding the RGB channels to identify black pixels
        # mask = cv2.bitwise_or(cv2.bitwise_or(b, g), r)
        # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # # Set the alpha channel values of black pixels to 0 (fully transparent)
        # a = cv2.bitwise_and(a, cv2.bitwise_not(mask))

        # # Merge the modified channels back into a 4-channel image
        # modified_image = cv2.merge((b, g, r, a))


        
        # mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)[1]
        # mask_inv = cv2.bitwise_not(mask)
        # b = cv2.bitwise_and(b, mask_inv)
        # g = cv2.bitwise_and(g, mask_inv)
        # r = cv2.bitwise_and(r, mask_inv)
        # modified_image = cv2.merge((b, g, r, a))
        return Image.from_numpy(modified_image)
    
    # def to_color(self, color):

    

    def paste(self, image: 'Image'):
        if self.size != image.size:
            image = image.resize(self.size)
        if not image.opacity:
            raise Exception('Image must have opacity')
        pil_img_copy = self.pil_img.copy()
        condition = image.np_arr[:,:,3] > 0
        condition = np.stack((condition,) * 3, axis=-1)
        output_image = np.where(condition, self.np_arr, image.np_arr[:,:,0:3])
        return Image.from_numpy(output_image)
        # pil_img_copy = self.add_alpha().pil_img
        # pil_img_copy.paste(image.pil_img, (0,0))
        # return Image.from_pil(pil_img_copy)
        

        # mask = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)[1]
        # mask_inv = cv2.bitwise_not(mask)
        # gray = self.gray().np_arr
        # _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # modified_image = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.uint8)
        # modified_image[:, :, 0:3] = self.np_arr
        # modified_image[:, :, 3] = mask
        # return Image.from_numpy(modified_image)
    
    # def add(self, image: 'Image'):
        # if self.size != image.size:
        #     image = imag



    # def add(self, image: 'Image'):





    def _numpy_to_pil(self) -> 'Image':
        # if self._np_arr is None:
        #     return self
        if self.opacity:
            self._pil_img = pimg.fromarray(cv2.cvtColor(self._np_arr, cv2.COLOR_BGRA2RGBA))
        else:   
            # self._pil_img = pimg.fromarray(cv2.cvtColor(self._np_arr, cv2.COLOR_BGR2RGB))
            self._pil_img = pimg.fromarray(self._np_arr)
        # 
        # self._pil_img = pimg.fromarray(cv2.cvtColor(self._np_arr, cv2.COLOR_BGRA2RGBA))
        return self


    def _pil_to_numpy(self) -> 'Image':
        # if self._pil_img is None:
        #     return self
        self._np_arr = np.array(self._pil_img)


    def _repr_png_(self):
        return self.pil_img._repr_png_()
    

    def resize(self, size=None, width: int = None, height: int=None, antialiazing=True) -> 'Image':
        # new_size = new_size if len(new_size) == 2 else (new_size[0], new_size[1])
        if size is None:
            if width is None:
                width = int(self.pil_img.width * height / self.pil_img.height)
            elif height is None:
                height = int(self.pil_img.height * width / self.pil_img.width)
        else:
            if len(size) == 3:
                width, height, _ = size
            else:
                width, height = size
            
        pil_img = self.pil_img.resize((width, height), pimg.ANTIALIAS if antialiazing else pimg.NEAREST)
        return Image.from_pil(pil_img)


    def resize_crop(self, size=(512, 512), crop='center'):
        image = self.pil_img
        w,h = image.size
        if w > h:
            w = int(w * size[1] / h)
            h = size[1]
            if isinstance('dsff', str):
                if crop=='center':
                    crop_start = (w - size[0]) // 2
                elif crop=='start':
                    crop_start = 0
                elif crop=='end':
                    crop_start = w - size[0]
            # im1 = im.crop((left, top, right, bottom))
            pil_resized_image = image.resize((w, h)).crop((crop_start, 0, crop_start + size[0], size[1]))
            return Image.from_pil(pil_resized_image)
        else:
            h = int(h * size[0] / w)
            w = size[0]
            if isinstance('dsff', str):
                if crop=='center':
                    crop_start = (h - size[1]) // 2
                elif crop=='start':
                    crop_start = 0
                elif crop=='end':
                    crop_start = h - size[1]
            pil_resized_image = image.resize((w, h)).crop((0, crop_start, size[0], crop_start + size[1]))
            return Image.from_pil(pil_resized_image)


    def sub_image(self, x, y, w, h):
        return self.pil_img.crop((x, y, x + w, y + h))


    def grid(images: List['Image'], cols=None, rows=None):
        # if cols is None:
        #     cols = len(images)
        if cols is None and rows is None:
            cols = min(len(images), 3)
            rows = int(np.ceil(len(images) / cols))
        if rows is None and cols is not None:
            rows = int(np.ceil(len(images) / cols))
        if cols is None and rows is not None:
            cols = int(np.ceil(len(images) / rows))

        
        return image_grid(images, rows=rows, cols=cols)

    def copy(self):
        img = Image.from_pil(self.pil_img)
        return img

    

    def __getitem__(self, key):
        sub_arr = self.np_arr[key]
        return Image.from_numpy(sub_arr)

        


    def color_hist(self, r=False, g=False, b=False, a=False, rgb=True, bar=True, bins=256):
        img = self.np_arr
        _r, _g, _b, _a = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
        r_hist = np.histogram(_r, bins=bins, range=(0,255))
        g_hist = np.histogram(_g, bins=bins, range=(0,255))
        b_hist = np.histogram(_b, bins=bins, range=(0,255))
        a_hist = np.histogram(_a, bins=bins, range=(0,255))
        plt.figure()
        plt.title("Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("Number of Pixels")
        x_dim = np.arange(0, bins)
        if r or rgb:
            if bar:
                plt.bar(x_dim, r_hist[0], color = 'r',label = 'Red')
            else:
                plt.plot(x_dim, r_hist[0], color = 'r',label = 'Red')
        if g or rgb:
            if bar:
                plt.bar(x_dim, g_hist[0], color = 'g',label = 'Green')
            else:
                plt.plot(x_dim, g_hist[0], color = 'g',label = 'Green')
        if b or rgb:
            if bar:
                plt.bar(x_dim, b_hist[0], color = 'b',label = 'Blue')
            else:
                plt.plot(x_dim, b_hist[0], color = 'b',label = 'Blue')
            
        if a:
            if bar:
                plt.bar(x_dim, a_hist[0], color = 'black',label = 'opacity')
            else:
                plt.plot(x_dim, a_hist[0], color = 'black',label = 'opacity')
        plt.legend()
        plt.xlim([0, bins])
        plt.show()




def pil_images_to_numpy(pil_images):
    images = [Image.from_pil(pil_image) for pil_image in pil_images]
    return np.array([image.np_arr for image in images])