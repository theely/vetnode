import click

from vetnode import cli


@click.group()
def main():
    pass

main.add_command(cli.diagnose)
main.add_command(cli.setup)


if __name__ == '__main__':
    main()