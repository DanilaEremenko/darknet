import json
import subprocess
import re
from pathlib import Path


class BoxResult():
    def __init__(self, left_x: int, top_y: int, width: int, height: int, class_name: str, class_prob: float):
        self.left_x = left_x
        self.top_y = top_y
        self.width = width
        self.height = height
        self.class_name = class_name
        self.class_prob = class_prob

    def to_dict(self) -> dict:
        return {
            "left_x": self.left_x,
            "top_y": self.top_y,
            "width": self.width,
            "height": self.height,
            "class_name": self.class_name,
            "class_prob": self.class_prob
        }


def get_box_from_str(line) -> BoxResult:
    class_str, coord_str = line.split("\\t")
    class_name, class_prob = class_str.split(":")
    class_prob = int(class_prob.replace("%", "")) / 100

    coord_str = re.sub(" +", " ", coord_str)
    box_args = coord_str \
        .replace("(", "") \
        .replace(")", "") \
        .replace(": ", ":") \
        .split(" ")

    return BoxResult(
        left_x=int(box_args[0].split(":")[1]),
        top_y=int(box_args[1].split(":")[1]),
        width=int(box_args[2].split(":")[1]),
        height=int(box_args[3].split(":")[1]),
        class_name=class_name,
        class_prob=class_prob
    )


def get_boxes(darknet_bin: str, data_path: str, arch_path: str, weights_path: str, images_path: str):
    args = [darknet_bin, 'detector', 'test', data_path, arch_path, weights_path, '-ext_output', '-thresh', '0.5']
    debug_str = f"{' '.join(args)} < {images_path}"
    print(debug_str)

    box_results = {}
    with open(images_path) as img_list_fp:
        for img_path in img_list_fp.read().split('\n'):
            if not Path(img_path).is_file():
                raise FileNotFoundError(f"img_path = {img_path} not exist")

    with open(images_path) as input_fp:
        res_str = subprocess.check_output(
            args=args,
            stderr=None,
            stdin=input_fp
        )
        # res_str = b" GPU isn't used \nmini_batch = 1, batch = 1, time_steps = 1, train = 0 \n\n seen 64, trained: 32013 K-images (500 Kilo-batches_64) \nEnter Image Path:  Detection layer: 82 - type = 28 \n Detection layer: 94 - type = 28 \n Detection layer: 106 - type = 28 \ndata/dog.jpg: Predicted in 12589.498000 milli-seconds.\nbicycle: 99%\t(left_x:  118   top_y:  124   width:  452   height:  309)\ndog: 100%\t(left_x:  124   top_y:  223   width:  196   height:  320)\ntruck: 93%\t(left_x:  474   top_y:   87   width:  216   height:   79)\nEnter Image Path:  Detection layer: 82 - type = 28 \n Detection layer: 94 - type = 28 \n Detection layer: 106 - type = 28 \ndata/eagle.jpg: Predicted in 11980.030000 milli-seconds.\nbird: 99%\t(left_x:  111   top_y:   89   width:  523   height:  353)\nEnter Image Path: "
        res_arr = str(res_str).split('\\n')
        for line in res_arr:
            if 'Predicted in' in line:
                img_path = line.split(':')[0]
                box_results[img_path] = []
            elif "left_x: " in line:
                box_results[img_path].append(get_box_from_str(line=line).to_dict())

    return box_results


def main():
    with open("my_config.json") as fp:
        cfg_data = json.load(fp)
        nn_cfg = cfg_data['nn']
        images_path = cfg_data['images_path']
        res_path = cfg_data['res_path']

        if not Path(res_path).parent.is_dir(): raise FileNotFoundError(f"parent dir of {res_path} not exists")

        for key, value in nn_cfg.items():
            if not Path(value).is_file():
                raise FileNotFoundError(f"{key} = {value}")

    box_results = get_boxes(**nn_cfg, images_path=images_path)

    with open(res_path, 'w') as fp:
        json.dump(fp=fp, obj={'results_map': box_results})


if __name__ == '__main__':
    main()
