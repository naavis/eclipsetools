import sys

import alignment
import utils.circlefinder
import utils.raw_reader
from preprocessing import preprocess_for_alignment


def main(args):
    ref_image = preprocess_for_alignment(utils.raw_reader.open_raw_image(args[1]))
    image_to_align = preprocess_for_alignment(utils.raw_reader.open_raw_image(args[2]))
    print(alignment.find_translation(ref_image, image_to_align))


if __name__ == '__main__':
    main(sys.argv)
