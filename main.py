import click

from eclipsetools.commands.align import align
from eclipsetools.commands.preprocess import preprocess_only
from eclipsetools.commands.stack import hdr_stack


@click.group(context_settings={"show_default": True})
def main():
    pass


main.add_command(align)  # type: ignore
main.add_command(preprocess_only)  # type: ignore
main.add_command(hdr_stack)  # type: ignore

if __name__ == "__main__":
    main()
