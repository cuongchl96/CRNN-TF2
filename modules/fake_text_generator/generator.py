import cv2
import os
import numpy as np 
import random

from modules.fake_text_generator.libgen.trdg.generators import (
    GeneratorFromStrings,
)

class FakeTextGenerator():
    def __init__(self, corpus, font_type='cmnd', text_size=35, skewing_angle=0, distorsion_orientation=3, blur=2):
        self.corpus = corpus
        self.font_type = font_type
        self.text_size = text_size
        self.skewing_angle = skewing_angle
        self.distorsion_orientation = distorsion_orientation
        self.blur = blur

    def gen(self, batch_size=16):
        samples = random.sample(self.corpus, k=batch_size)
        generator = GeneratorFromStrings(
            samples,
            count=len(samples),
            blur=self.blur,
            random_blur=True,
            size=self.text_size,
            skewing_angle=self.skewing_angle,
            random_skew=True,
            distorsion_orientation=self.distorsion_orientation,
            language=self.font_type)

        images = []
        labels = []
        for image, label in generator:
            image = np.array(image)[..., ::-1]
            images.append(image)
            labels.append(label)

        return images, labels

if __name__ == "__main__":
    corpus = []
    for i in range(10000):
        seq = ''
        for j in range(9):
            seq += str(random.randint(0, 9))
        corpus.append(seq)

    generator = FakeTextGenerator(corpus=corpus)

    while True:
        generator.gen(batch_size=16)
