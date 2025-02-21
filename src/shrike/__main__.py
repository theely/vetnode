import click

from shrike import cli


@click.group()
def main():
    pass

main.add_command(cli.diagnose)


if __name__ == '__main__':
    main()