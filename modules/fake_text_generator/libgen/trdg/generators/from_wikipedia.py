from modules.fake_text_generator.libgen.trdg.generators.from_strings import GeneratorFromStrings
from modules.fake_text_generator.libgen.trdg.data_generator import FakeTextDataGenerator
from modules.fake_text_generator.libgen.trdg.string_generator import create_strings_from_wikipedia
from modules.fake_text_generator.libgen.trdg.utils import load_dict, load_fonts

class GeneratorFromWikipedia:
    """Generator that uses sentences taken from random Wikipedia articles"""

    def __init__(
        self,
        count=-1,
        minimum_length=1,
        fonts=[],
        language="en",
        size=32,
        skewing_angle=0,
        random_skew=False,
        blur=0,
        random_blur=False,
        background_type=0,
        distorsion_type=0,
        distorsion_orientation=0,
        is_handwritten=False,
        width=-1,
        alignment=1,
        text_color="#282828",
        orientation=0,
        space_width=1.0,
        character_spacing=0,
        margins=(5, 5, 5, 5),
        fit=False,
        output_mask=False,
    ):
        self.count = count
        self.minimum_length = minimum_length
        self.language = language
        self.generator = GeneratorFromStrings(
            create_strings_from_wikipedia(self.minimum_length, 1000, self.language),
            count,
            fonts if len(fonts) else load_fonts(language),
            language,
            size,
            skewing_angle,
            random_skew,
            blur,
            random_blur,
            background_type,
            distorsion_type,
            distorsion_orientation,
            is_handwritten,
            width,
            alignment,
            text_color,
            orientation,
            space_width,
            character_spacing,
            margins,
            fit,
            output_mask,
        )

    def __iter__(self):
        return self.generator

    def __next__(self):
        return self.next()

    def next(self):
        if self.generator.generated_count >= 999:
            self.generator.strings = create_strings_from_wikipedia(
                self.minimum_length, 1000, self.language
            )
        return self.generator.next()
