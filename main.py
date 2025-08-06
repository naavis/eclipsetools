import click

from eclipsetools.commands.align import align
from eclipsetools.commands.filter import filter_group
from eclipsetools.commands.stack import stack
from eclipsetools.commands.utils import utils


@click.group(context_settings={"show_default": True})
def main():
    pass


main.add_command(align)  # type: ignore
main.add_command(filter_group)  # type: ignore
main.add_command(stack)  # type: ignore
main.add_command(utils)  # type: ignore

if __name__ == "__main__":
    main()
