from pyIMS.utils.stations import get_ortho_img
from pyIMS.utils.io import read_ply
from PIL import Image
import json



if __name__ == "__main__":
    points, colors  = read_ply('../pyIMS/data/qiyu/street0_1/pc/pc.ply')
    img, _, info = get_ortho_img(points, colors, view= 'top')
    img = Image.fromarray(img)
    img.save('./pc_top.png')
    with open('./pc_top_info.json', 'w') as f:
        json.dump(info, f, indent=4)

    img, _, info = get_ortho_img(points, colors, view= 'front')
    img = Image.fromarray(img)
    img.save('./pc_front.png')
    with open('./pc_front_info.json', 'w') as f:
        json.dump(info, f, indent=4)
