import sys
from models.enhanced_resnet import EnhancedResnet
from utils.model_trainer import train_model
from utils.denoiser_trainer import train_denoiser
from data.data_utils import get_data


def trainer(mode = "complete",epochs=20,batch_size=8,data_size="medium"):

    model = EnhancedResnet()
    	
    #try:
    #    assert mode in ["complete","denoiser","model"]
    #except:
    #    print("\nERROR: Invalid train mode. Training on both Denoiser and Resnet models.")


    if mode == "model":
        tr,tt = get_data(data_size)
        ans = train_model(model.residualnet,epochs,batch_size,tr,tt)
    elif mode == "denoiser":
        tr,tt = get_data(data_size,False)
        ans = train_denoiser(model.denoised_layer,epochs,batch_size,tr,tt)
    else:
        i = 0
        tr1,tt1 = get_data(data_size)
        tr2,tt2 = get_data(data_size,False)
        while i <= epochs:
            tr,tt = get_data(data_size,False)
            tr,tt = get_data(data_size,False)
            ans = train_model(model.residualnet,4,batch_size,tr,tt)
            if ans == "end":
                sys.exit()
            ans = train_denoiser(model.denoised_layer,4,batch_size,tr,tt)
            if ans == "end":
                sys.exit()
            i = i+4

			
if __name__ == '__main__':

    #model = sys.argv[1]
    #layers = sys.argv[1]
    #epochs = sys.argv[2]
    #batches = sys.argv[3]
    #datasize = sys.argv[4]
	
    '''
	# MODE = "denoiser" for training denoised_layer, "model" for training Resnet layers, and "complete" for both
	
	# DATA_SIZE = "small" for 15000 training and 3000 testing images, "medium" for 25000 training and 5000 testing images
	  This can be changed in data_utils.py file in data folder. Change the values in "subsetting" method call.
	'''
	
    trainer(mode = "denoiser",epochs = 10,batch_size = 4,data_size = "small")
	
