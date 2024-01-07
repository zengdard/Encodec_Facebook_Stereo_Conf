from transformers import EncodecModel, AutoProcessor, EncodecConfig, EncodecFeatureExtractor
import torchaudio
from torchaudio.transforms import Resample
from encodec.utils import convert_audio



#Télécharger une fois les modèles +  process
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")


model.save_pretrained('/MODEL')
processor.save_pretrained('/MODEL')


#Charger votre audio
audio_file_path = 'HTR.wav'
wav, sr = torchaudio.load(audio_file_path)
new_sample_rate = 24000  #
# Créer une instance de la transformation de resampling
resample_transform = Resample(sr, new_sample_rate) 
# Appliquer la transformation de resampling au signal audio car Model Encodec 24khz dispo en 48khz sur HuggincFace
wav = resample_transform(wav)


encoder = EncodecFeatureExtractor(feature_size=2)
configuration = EncodecConfig(audio_channels=2)
model = EncodecModel(configuration)
#Preprocess
inputs = encoder.__call__(raw_audio=wav, return_tensors="pt", sampling_rate=new_sample_rate, padding=True)

outputs = model(**inputs)
audio_codes = outputs.audio_codes
audio_values = outputs.audio_values

print(audio_codes, audio_values)

#EncodecOutput(audio_codes=tensor([[[[0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0]]]]), audio_values=tensor([[[-0.1701, -0.1270, -0.1641,  ..., -0.1541, -0.0950, -0.1555],
#         [-0.2411, -0.2813, -0.2685,  ..., -0.2408, -0.2893, -0.2432]]],
#      grad_fn=<SliceBackward0>))
