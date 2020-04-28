
from textsimilarity.utils.utils import *

def main(config):

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        print(
        "Output directory ({}) already exists and is not empty. ".format(config.output_dir)
        print("Do you want overwrite it? type y or n")
        overwrite = input()
        if overwrite = 'n':
            return
    )

    # device ready
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])

    # set random seed
    set_seed(config)

    # load 



