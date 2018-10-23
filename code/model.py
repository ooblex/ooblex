from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import *
from keras.applications import *

from pixel_shuffler import PixelShuffler

optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

IMAGE_SHAPE = (256,256,3)

def conv( filters ):
    def block(x):
        x = Conv2D( filters, kernel_size=3, strides=2, kernel_initializer=RandomNormal(0, 0.02), use_bias=False, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale( filters ):
    def block(x):
        x = Conv2D(filters*4, kernel_size=5, use_bias=False, kernel_initializer=RandomNormal(0, 0.02), padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Encoder():
	input_ = Input( shape=IMAGE_SHAPE )
	x = input_
	x = conv(  32)(x)
	x = conv(  64)(x)
	x = conv( 128)(x)
	x = conv( 256)(x)
	x = conv( 512)(x)
	x = conv(1024)(x)
	x = conv(2048)(x)
	x = Dense( 1024*2 )( Flatten()(x) )
	x = Dense(4*4*1024*2)(x)
	x = Reshape((4,4,1024*2))(x)
	x = upscale(512*2)(x)
	return Model( input_, x )
	
def Encoder0():
	inp = Input(shape=IMAGE_SHAPE)
	x = Conv2D(64, kernel_size=5, kernel_initializer=RandomNormal(0, 0.02), use_bias=False, padding="same")(inp)
	x = conv(128)(x)
	x = conv(256)(x)
	x = conv(512) (x)
	x = conv(1024)(x)
	x = Dense(1024)(Flatten()(x))
	x = Dense(4*4*1024)(x)
	x = Reshape((4, 4, 1024))(x)
	out = upscale(512)(x)
	return Model(inputs=inp, outputs=out)

def Decoder():
	input_ = Input( shape=(8,8,512*2) )
	x = input_
	x = upscale(256)(x)
	x = upscale(128)(x)
	x = upscale( 64)(x)

	xi = x
	x = Conv2D( 64, kernel_size=3, padding='same', kernel_initializer=RandomNormal(0, 0.02), activation='sigmoid', dilation_rate= 2)(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D( 64, kernel_size=3, padding='same', kernel_initializer=RandomNormal(0, 0.02), activation='sigmoid', dilation_rate= 2)(x)
	x = add([x, xi])

	# out64 = Conv2D(64, kernel_size=3, padding='same')(x)
	# out64 = LeakyReLU(alpha=0.1)(out64)
	# out64 = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(out64)

	x = upscale(32)(x)
	x = upscale(16)(x)

	xi = x
	x = Conv2D( 16, kernel_size=3, padding='same', kernel_initializer=RandomNormal(0, 0.02), activation='sigmoid', dilation_rate= 1)(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D( 16, kernel_size=3, padding='same', kernel_initializer=RandomNormal(0, 0.02), activation='sigmoid', dilation_rate= 1)(x)
	x = add([x, xi])
	
	xi = x
	x = Conv2D( 16, kernel_size=3, padding='same', kernel_initializer=RandomNormal(0, 0.02), activation='sigmoid', dilation_rate= 1)(x)
	x = LeakyReLU(alpha=0.2)(x)
	x = Conv2D( 16, kernel_size=3, padding='same', kernel_initializer=RandomNormal(0, 0.02), activation='sigmoid', dilation_rate= 1)(x)
	x = add([x, xi])
	
	

	#alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
	rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
	#out = concatenate([rgb,alpha])
	
	return Model( input_, rgb )
	

encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

encoder_swift = Encoder()
decoder_A_swift = Decoder()


x = Input( shape=IMAGE_SHAPE )

autoencoder_A = Model( x, decoder_A( encoder(x) ) )
autoencoder_B = Model( x, decoder_B( encoder(x) ) )
autoencoder_A.compile( optimizer=optimizer, loss='mean_absolute_error' )
autoencoder_B.compile( optimizer=optimizer, loss='mean_absolute_error' )

autoencoder_A_swift = Model( x, decoder_A_swift( encoder_swift(x) ) )
autoencoder_A_swift.compile( optimizer=optimizer, loss='mean_absolute_error' )



