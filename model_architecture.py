import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

def build_cnn_lstm_model(vocab_size, max_length):
    """
    Builds the CNN + LSTM model for image captioning.
    Architecture:
    1. Feature Extractor (CNN like ResNet50): Extracts 2048-dim features from images.
    2. Sequence Processor (LSTM): Processes partial text sequences translated into word embeddings.
    3. Decoder: Combines CNN features and LSTM hidden states to predict the next word.
    """
    
    # ==========================================
    # 1. Feature Extractor (CNN)
    # ==========================================
    # Load ResNet50 trained on ImageNet, without the top classification layers
    cnn_base = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    
    # Freeze the layers of CNN so we don't destroy pre-trained weights during initial training
    for layer in cnn_base.layers:
        layer.trainable = False
        
    # Image Input layer (ResNet outputs a 2048-element vector with avg pooling)
    image_input = Input(shape=(2048,), name="image_features_input")
    fe1 = Dropout(0.5)(image_input)
    # Reduce dimensionality to 256
    fe2 = Dense(256, activation='relu', name="cnn_dense")(fe1) 
    
    # ==========================================
    # 2. Sequence Processor (RNN/LSTM)
    # ==========================================
    # Partial Caption Input layer
    sequence_input = Input(shape=(max_length,), name="text_sequence_input")
    # Embedding layer (vocab_size -> 256 dim vectors)
    se1 = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(sequence_input)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, name="lstm_layer")(se2)
    
    # ==========================================
    # 3. Decoder Mode (Combine both vectors)
    # ==========================================
    # Merge the features from the CNN and the LSTM
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    
    # Output layer predicting the next word in the vocabulary
    outputs = Dense(vocab_size, activation='softmax', name="word_prediction")(decoder2)
    
    # Compile Model
    # Tie inputs and outputs together
    model = Model(inputs=[image_input, sequence_input], outputs=outputs, name="Image_Captioning_Model")
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

if __name__ == "__main__":
    # Example for Flickr8k dataset specs
    VOCAB_SIZE = 8300 # Approx unique words in Flickr8k after cleaning
    MAX_LENGTH = 34   # Maximum length of caption sequence in Flickr8k
    
    print(f"Building CNN-LSTM Image Captioning Model...")
    print(f"Vocabulary Size: {VOCAB_SIZE}, Maximum Sequence Length: {MAX_LENGTH}")
    
    model = build_cnn_lstm_model(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
    
    # Print the network blueprint
    model.summary()
    print("\nModel built successfully! You can visualize or train it.")
    
    # To save this model structure (without weights)
    # model.save('model.h5')
