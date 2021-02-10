from unet_model import basic_model, tl_model
from Datagen import DataGen, root_mean_squared_error, train_feed_data
from utils import call_backs, save_on_epoch_end
import matplotlib.pyplot as plt
# import da


def main(model, train_id,image_size,train_path,batch_size,epochs,valid_split=.2):
    unet_model = model(image_size=256)
    train_gen, valid_gen, train_steps, valid_steps = train_feed_data(image_size = image_size,
                                                                     train_path = train_path,
                                                                     batch_size = batch_size,
                                                                     valid_split=valid_split)

    callback = call_backs(train_id=train_id)
    unet_model.fit_generator(train_gen,
                             validation_data=valid_gen,
                             steps_per_epoch=train_steps,
                             validation_steps=valid_steps,
                             epochs=epochs, callbacks=callback)
    save_on_epoch_end(train_id=train_id, model=unet_model)


if __name__ == '__main__':
    image_size = 256
    train_path = './Data/Train_Data_new'
    # train_path = r"D:\Documents\RL\Projects\Sealion\Elephant\Data\Elephant"
    batch_size = 4
    epochs = 50
    main(model = tl_model,
        train_id='tl_model_train5_small1',
         image_size=image_size,
         train_path=train_path,
         batch_size=batch_size,
         epochs=epochs,
         valid_split=.2)
