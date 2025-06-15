import sys

import eclipsetools.utils.raw_reader
from eclipsetools import alignment
from eclipsetools.preprocessing import preprocess_for_alignment


def main(args):
    ref_image = preprocess_for_alignment(
        eclipsetools.utils.raw_reader.open_raw_image(args[1]).data)
    image_to_align = preprocess_for_alignment(
        eclipsetools.utils.raw_reader.open_raw_image(args[2]).data)
    print(alignment.find_translation(ref_image, image_to_align))


if __name__ == '__main__':
    main(sys.argv)
