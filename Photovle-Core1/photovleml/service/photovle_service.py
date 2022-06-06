import os
from .train import trainer
from .predict import predictor, video_predictor


class PhotovleService:
    @staticmethod
    def train(user_id):
        trainer(user_id)
    
    @staticmethod
    def predict(user_id):
        return predictor(user_id)

    @staticmethod
    def predict_video(user_id):
        return video_predictor(user_id)
