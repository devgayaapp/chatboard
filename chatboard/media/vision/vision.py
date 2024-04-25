
import json
from components.image.image import BoundingBox, Image
from components.image.llava_client import LlavaClient
from enum import Enum, EnumMeta
import re
from pydantic import BaseModel

BOOL_RESPONSE = "answer yes/no"
INT_RESPONSE = "answer with a number only"
# BOUNDING_BOX_RESPONSE = " use the following format [x1, y1, x2, y2]. repeat for multiple boxes"
BOUNDING_BOX_RESPONSE = " use the following format [x1, y1, x2, y2]. return a single box"

boundingbox_regex_pattern = r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]"


class VisionException(Exception):
    pass
class Vision:

    def __init__(self, client=None):
        if client is None:
            client = LlavaClient()
        self.client = client

    async def complete(self, prompt: str, img: Image, output_type=str, output: str=None):

        # if type(output_type) == EnumMeta:
        #     output_type = output_type.__members__.values()
        #     values = [v.value for v in output_type]
        #     bool_text = await self.client.complete(prompt + BOOL_RESPONSE, img)
        if output_type == bool:
            bool_text = await self.client.complete(prompt + BOOL_RESPONSE, img)
            bool_text = bool_text.strip().lower().strip('"').strip("'")
            if 'yes' in bool_text or 'true' in bool_text:
                return True
            elif 'no' in bool_text or 'false' in bool_text:
                return False
        elif output_type == BoundingBox or output == 'box':
            bb_text = await self.client.complete(prompt + BOUNDING_BOX_RESPONSE, img)
            print(bb_text)
            try:                
                matches = re.findall(boundingbox_regex_pattern, bb_text)
                bounding_boxes = [BoundingBox(point1=(float(match[0]), float(match[1])), point2=(float(match[2]), float(match[3]))) for match in matches]
                return bounding_boxes
            except Exception as e:
                raise VisionException(f"Bounding box decoding Error: {bb_text}")            
            #     coords = json.loads(bb_text)
            #     return BoundingBox(point1=coords[:2], point2=coords[2:])
            # except json.JSONDecodeError:
            #     raise VisionException(f"Bounding box decoding Error: {bb_text}")            
        elif output_type == str:
            text = await self.client.complete(prompt, img)
            text = text.strip().strip('"').strip("'")
            return text
        elif output_type == int:
            int_text = await self.client.complete(prompt + INT_RESPONSE, img)
            int_text = int_text.strip().strip('"').strip("'")
            try:
                return int(int_text)
            except ValueError:
                return None
        return text
