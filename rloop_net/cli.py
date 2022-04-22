import click

@click.command()
@click.option('--epochs', default=1, help='Number of epochs')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(epochs, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(epochs):
        click.echo(f"Hello {name}!")

if __name__ == '__main__':
    hello()