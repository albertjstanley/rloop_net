import click 

from rloop_net.data import generate_example_batch_input, generate_example_batch_output, generate_example_batch_output
from rloop_net.model import get_rloop_model
from rloop_net.utils import get_unique_run_id
from tensorflow.keras.optimizers import Adam

from keras.callbacks import CSVLogger
from pathlib import Path

RUN_DIRECTORY = Path(get_unique_run_id())

def get_callbacks():
    csv_logger = CSVLogger(RUN_DIRECTORY / 'log.csv', append=True, separator=',')
    return [csv_logger]

@click.command()
@click.option('--dir', required=True, help='Path to save model to')
@click.option('--num_epochs', default=5)
@click.option('--lr', default=0.0001)
@click.option('--counts_weight', default=10000)
def example_train(dir, num_epochs, lr, counts_weight):

    import os
    os.mkdir(RUN_DIRECTORY)

    batch_size = 1
    model = get_rloop_model()
    x = generate_example_batch_input()
    y = generate_example_batch_output()

    # Compile model
    model.compile(loss=["mse","mse"],optimizer=Adam(lr),loss_weights=[1, counts_weight])

    history = model.fit(x=x,y=y, 
                        epochs=num_epochs, 
                        batch_size=batch_size,
                        shuffle=False,
                        callbacks=get_callbacks())
    model.save(RUN_DIRECTORY / "model")



if __name__=="__main__":
    example_train()