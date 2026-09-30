from tensorflow.keras.models import load_model

model = load_model("hybrid_cnn_vgg16_final.h5", compile=False)
model.save("hybrid_cnn_vgg16_final.keras")
print(" model saved")
