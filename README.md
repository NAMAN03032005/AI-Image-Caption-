# AI Image Caption Generator - Backend

This folder contains the Flask API and Deep Learning model architecture for the Image Caption Generator project.

## Project Files

- `app.py`: The main Flask REST API. It handles file uploads, processes the images, and returns the generated captions. Out-of-the-box it uses a highly capable pre-trained model (BLIP) from Hugging Face for best results.
- `model_architecture.py`: Contains the TensorFlow/Keras code to build a custom CNN (`ResNet50`) + RNN (`LSTM`) model suitable for training from scratch on the Flickr8k or Flickr30k datasets.
- `requirements.txt`: Python package dependencies.
- `static/uploads/`: Directory where uploaded user images are temporarily saved for processing.

## How to Run the Backend

1. **Install Dependencies**
   Make sure you have Python 3.8+ installed. Open a terminal in the `backend/` folder and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask Server**
   ```bash
   python app.py
   ```
   The backend will start running on `http://127.0.0.1:5000/`. Wait for the console to indicate that the model has loaded successfully.

3. **API Endpoint**
   The frontend connects to `POST http://127.0.0.1:5000/predict` and sends a `multipart/form-data` with an image file under the `image` key.

## Training Custom CNN + LSTM (Optional)

If you'd like to train your own custom ResNet50+LSTM model:
1. Download [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k).
2. Look at the `model_architecture.py` blueprint and train it with standard NLP techniques (vectorizing captions, categorical crossentropy, etc).
3. Save the trained model to `model.h5`.
4. Update `app.py` to `load_model('model.h5')`.
