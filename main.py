import click

from eclipsetools.commands.align import align
from eclipsetools.commands.filter import unsharp_mask_filter
from eclipsetools.commands.preprocess import preprocess_only
from eclipsetools.commands.stack import hdr_stack, average_stack
from eclipsetools.commands.utils import find_moon, create_moon_mask, log_stretch


@click.group(context_settings={"show_default": True})
def main():
    pass


main.add_command(align)  # type: ignore
main.add_command(preprocess_only)  # type: ignore
main.add_command(hdr_stack)  # type: ignore
main.add_command(average_stack)  # type: ignore
main.add_command(find_moon)  # type: ignore
main.add_command(unsharp_mask_filter)  # type: ignore
main.add_command(create_moon_mask)  # type: ignore
main.add_command(log_stretch)  # type: ignore

if __name__ == "__main__":
    main()
