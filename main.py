import click

from eclipsetools.commands.align import align
from eclipsetools.commands.filter import filter
from eclipsetools.commands.preprocess import preprocess_only
from eclipsetools.commands.stack import stack
from eclipsetools.commands.utils import utils


@click.group(context_settings={"show_default": True})
def main():
    pass


main.add_command(align)  # type: ignore
main.add_command(preprocess_only)  # type: ignore
main.add_command(filter)  # type: ignore
main.add_command(stack)  # type: ignore
main.add_command(utils)  # type: ignore

if __name__ == "__main__":
    main()
